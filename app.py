# app.py
import io
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import HORIZON, LOOKBACK, TARGET_COLUMNS
from src.inference import predict_future

APP_NAME = "Rail Transit Time-Series Forecast API"
API_KEY = os.getenv("API_KEY", "").strip()

# Callback 协议格式已经通过官网真实接口验证，固定不再改变。
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "20"))
CALLBACK_RETRIES = int(os.getenv("CALLBACK_RETRIES", "5"))

# 关键：官网 callback 接口在并发写入时会出现 pending 计数不下降的现象。
# 因此回调严格串行发送，避免多个 callback 同时到达平台官网。
CALLBACK_WORKERS = 1

# 每个回调至少在对应 /predict 请求入队这么久之后再发送，
# 保证官网先完成原请求登记。使用“最小年龄”而不是每条固定 sleep，
# 因此排队中的后续回调不会重复等待 1 秒。
CALLBACK_MIN_AGE = max(
    0.0, float(os.getenv("CALLBACK_MIN_AGE", "1.0"))
)

# 两次 callback POST 之间保留一个很小的间隔，进一步避免官网后端写入竞争。
CALLBACK_GAP = max(
    0.0, float(os.getenv("CALLBACK_GAP", "0.25"))
)

CALLBACK_EXECUTOR = ThreadPoolExecutor(
    max_workers=CALLBACK_WORKERS,
    thread_name_prefix="evaluation-callback-serial",
)

app = FastAPI(title=APP_NAME, version="2.6.0")


def ok(body: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=200, content=body)


def fail(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"code": -1, "message": str(message), "predictions": []},
    )


def check_api_key(request: Request) -> Optional[JSONResponse]:
    if not API_KEY:
        return None
    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return JSONResponse(
            status_code=401,
            content={"code": -1, "message": "unauthorized", "predictions": []},
        )
    return None


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def download_history(history_url: str, target_cols: List[str]) -> List[Dict[str, Any]]:
    if not history_url:
        raise ValueError("history 为空，同时 history_url 也为空")

    resp = requests.get(history_url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"history_url CSV 缺少字段: {missing}")

    return [
        {
            "step": step,
            "values": {c: safe_float(row[c]) for c in target_cols},
        }
        for step, (_, row) in enumerate(df.iterrows())
    ]


def history_to_df(history: Any, history_url: str, target_cols: List[str]) -> pd.DataFrame:
    if not history:
        history = download_history(history_url, target_cols)

    if not isinstance(history, list):
        raise ValueError("history 必须是数组")

    if history and all(isinstance(x, dict) and "step" in x for x in history):
        history = sorted(history, key=lambda x: int(x["step"]))

    rows = []
    for item in history:
        if isinstance(item, dict):
            source = item["values"] if isinstance(item.get("values"), dict) else item
            rows.append({c: safe_float(source.get(c)) for c in target_cols})
        elif isinstance(item, (list, tuple)):
            rows.append(
                {
                    c: safe_float(item[i]) if i < len(item) else np.nan
                    for i, c in enumerate(target_cols)
                }
            )
        else:
            raise ValueError(f"history 数据类型不支持: {type(item).__name__}")

    if not rows:
        raise ValueError("history 没有有效数据")

    df = pd.DataFrame(rows, columns=target_cols).reindex(columns=TARGET_COLUMNS)

    if len(df) < LOOKBACK:
        raise ValueError(f"历史数据不足: 实际 {len(df)}，至少需要 {LOOKBACK}")

    return df


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(payload.get("requestId", "")).strip()
    if not request_id:
        raise ValueError("缺少 requestId")

    horizon = int(payload.get("forecast_horizon", HORIZON))
    if horizon != HORIZON:
        raise ValueError(f"forecast_horizon 必须为 {HORIZON}，实际为 {horizon}")

    target_cols = payload.get("target_columns", TARGET_COLUMNS)
    if not isinstance(target_cols, list):
        raise ValueError("target_columns 必须是数组")

    if len(target_cols) != len(TARGET_COLUMNS) or set(target_cols) != set(TARGET_COLUMNS):
        raise ValueError(f"target_columns 必须完整包含: {TARGET_COLUMNS}")

    return {
        "request_id": request_id,
        "target_cols": target_cols,
        "history_length": int(payload.get("history_length", 512)),
    }


def repair_pred(pred: np.ndarray, history_df: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    expected = (HORIZON, len(TARGET_COLUMNS))

    if pred.shape != expected:
        raise ValueError(f"模型输出 shape 错误: 实际 {pred.shape}，要求 {expected}")

    last_values = history_df[TARGET_COLUMNS].iloc[-1].to_numpy(dtype=np.float64)
    for j in range(pred.shape[1]):
        bad = ~np.isfinite(pred[:, j])
        if np.any(bad):
            fallback = last_values[j] if np.isfinite(last_values[j]) else 0.0
            pred[bad, j] = fallback

    return pred


def build_predictions(pred: np.ndarray, target_cols: List[str]) -> List[Dict[str, Any]]:
    col_index = {name: i for i, name in enumerate(TARGET_COLUMNS)}
    return [
        {
            "step": step,
            "values": {
                col: float(pred[step, col_index[col]])
                for col in target_cols
            },
        }
        for step in range(HORIZON)
    ]


def run_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = validate_payload(payload)

    history_df = history_to_df(
        payload.get("history", []),
        str(payload.get("history_url", "") or ""),
        meta["target_cols"],
    )

    if len(history_df) != meta["history_length"]:
        print(
            f"[WARN] requestId={meta['request_id']} "
            f"history_length={meta['history_length']} actual={len(history_df)}"
        )

    pred = repair_pred(predict_future(history_df), history_df)
    predictions = build_predictions(pred, meta["target_cols"])

    return {
        "requestId": meta["request_id"],
        "code": 0,
        "message": "success",
        "predictions": predictions,
    }


def build_callback_payload(
    request_id: str,
    callback_token: Any,
    code: int,
    message: str,
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    已通过官网真实接口验证的 callback 格式，后续固定保持：

    {
        "callback_token": "...",
        "results": [
            {
                "request_id": "...",
                "data": {
                    "code": 0,
                    "message": "success",
                    "predictions": [...]
                }
            }
        ]
    }

    顶层兼容字段继续保留，因为官网已经验证可接受。
    """
    business_data = {
        "code": int(code),
        "message": str(message),
        "predictions": predictions,
    }

    return {
        "requestId": request_id,
        "callback_token": callback_token,
        "results": [
            {
                "request_id": request_id,
                "data": business_data,
            }
        ],
        "code": int(code),
        "message": str(message),
        "predictions": predictions,
    }


def send_callback(
    callback_url: str,
    body: Dict[str, Any],
    queued_at: float,
) -> None:
    request_id = str(body.get("requestId", "") or "")

    # 只保证“入队后至少等待 CALLBACK_MIN_AGE”，不是每条固定等待。
    age = time.monotonic() - queued_at
    wait_seconds = max(0.0, CALLBACK_MIN_AGE - age)
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    for attempt in range(1, CALLBACK_RETRIES + 1):
        try:
            resp = requests.post(
                callback_url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                timeout=(5.0, CALLBACK_TIMEOUT),
            )

            response_text = (resp.text or "").replace("\n", " ")[:500]

            if 200 <= resp.status_code < 300:
                print(
                    f"[CALLBACK OK] requestId={request_id} "
                    f"status={resp.status_code} attempt={attempt} "
                    f"response={response_text!r}"
                )
                if CALLBACK_GAP > 0:
                    time.sleep(CALLBACK_GAP)
                return

            error = f"HTTP {resp.status_code}: {response_text}"
        except Exception as exc:
            error = repr(exc)

        print(
            f"[CALLBACK RETRY] requestId={request_id} "
            f"{attempt}/{CALLBACK_RETRIES}: {error}"
        )

        if attempt < CALLBACK_RETRIES:
            time.sleep(min(0.5 * attempt, 2.0))

    print(f"[CALLBACK FAILED] requestId={request_id}")
    if CALLBACK_GAP > 0:
        time.sleep(CALLBACK_GAP)


def submit_callback(callback_url: str, body: Dict[str, Any]) -> None:
    request_id = str(body.get("requestId", "") or "")
    queued_at = time.monotonic()
    try:
        CALLBACK_EXECUTOR.submit(
            send_callback,
            callback_url,
            body,
            queued_at,
        )
        print(
            f"[CALLBACK QUEUED] requestId={request_id} "
            f"serial=true min_age={CALLBACK_MIN_AGE:.2f}s "
            f"gap={CALLBACK_GAP:.2f}s"
        )
    except Exception as exc:
        print(f"[CALLBACK QUEUE FAILED] requestId={request_id}: {exc!r}")


@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "status": "ok",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.6.0",
        "lookback": LOOKBACK,
        "forecast_horizon": HORIZON,
        "target_columns": TARGET_COLUMNS,
        "callback_timeout": CALLBACK_TIMEOUT,
        "callback_retries": CALLBACK_RETRIES,
        "callback_workers": CALLBACK_WORKERS,
        "callback_serial": True,
        "callback_min_age": CALLBACK_MIN_AGE,
        "callback_gap": CALLBACK_GAP,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any], request: Request):
    auth_error = check_api_key(request)
    if auth_error is not None:
        return auth_error

    try:
        if payload.get("batch") is True:
            return fail("当前联调版本要求 batch_size=1")

        result = run_one(payload)

        callback_url = str(payload.get("callback_url", "") or "")
        if callback_url:
            callback_payload = build_callback_payload(
                request_id=result["requestId"],
                callback_token=payload.get("callback_token"),
                code=0,
                message="success",
                predictions=result["predictions"],
            )
            submit_callback(callback_url, callback_payload)

        return ok(
            {
                "code": 0,
                "message": "success",
                "predictions": result["predictions"],
            }
        )

    except Exception as exc:
        request_id = str(payload.get("requestId", "") or "")
        print(f"[PREDICT ERROR] requestId={request_id}: {exc}")
        traceback.print_exc()

        callback_url = str(payload.get("callback_url", "") or "")
        if callback_url and request_id:
            callback_payload = build_callback_payload(
                request_id=request_id,
                callback_token=payload.get("callback_token"),
                code=-1,
                message=str(exc),
                predictions=[],
            )
            submit_callback(callback_url, callback_payload)

        return fail(str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8800")),
        workers=1,
        log_level="info",
    )
