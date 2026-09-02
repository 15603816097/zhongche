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
from src.inference import LGB_INFER_THREADS, load_models, predict_future

APP_NAME = "Rail Transit Time-Series Forecast API"
APP_VERSION = "2.9.0"
API_KEY = os.getenv("API_KEY", "").strip()

# -----------------------------------------------------------------------------
# 官方异步回调协议
# -----------------------------------------------------------------------------
# callback JSON 结构已经通过官网真实接口验证，绝对不要随意修改。
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "20"))
CALLBACK_RETRIES = int(os.getenv("CALLBACK_RETRIES", "5"))

# 官网 callback 后端并发写入曾出现 pending 计数竞争，因此回调继续严格串行。
CALLBACK_WORKERS = 1

# 从“收到 /predict 请求”的时刻开始计算最小等待年龄。
# 如果模型推理本身已经超过该时间，callback 不会额外等待。
CALLBACK_MIN_AGE = max(
    0.0, float(os.getenv("CALLBACK_MIN_AGE", "1.0"))
)

# 连续 callback 之间保留小间隔，规避官网后端并发写入竞争。
CALLBACK_GAP = max(
    0.0, float(os.getenv("CALLBACK_GAP", "0.25"))
)

CALLBACK_EXECUTOR = ThreadPoolExecutor(
    max_workers=CALLBACK_WORKERS,
    thread_name_prefix="evaluation-callback-serial",
)

# -----------------------------------------------------------------------------
# 异步预测执行器
# -----------------------------------------------------------------------------
# 2.8.0 使用 1 个 worker，单条约 16.6 秒时 50 条总耗时约 14 分钟，
# 会超过官网整批异步等待窗口。
# 2.9.0 默认使用 2 个外层 worker；内部 LightGBM 改为共享线程池的稀疏预测，
# 不再触发 MultiOutputRegressor 的进程级 joblib 预测。
PREDICT_WORKERS = max(1, int(os.getenv("PREDICT_WORKERS", "2")))
PREDICT_EXECUTOR = ThreadPoolExecutor(
    max_workers=PREDICT_WORKERS,
    thread_name_prefix="evaluation-predict-worker",
)

MODEL_STATUS: Dict[str, Any] = {
    "loaded": False,
    "ensemble_version": None,
    "validation_rmse": None,
    "validation_direction_accuracy": None,
}

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.on_event("startup")
def preload_latest_models() -> None:
    """服务启动时一次性预加载最终模型与融合权重。"""
    started = time.perf_counter()
    _, _, _, _, ensemble_config = load_models()

    MODEL_STATUS["loaded"] = True
    MODEL_STATUS["ensemble_version"] = ensemble_config.get("version")
    MODEL_STATUS["validation_rmse"] = ensemble_config.get("validation_rmse")
    MODEL_STATUS["validation_direction_accuracy"] = ensemble_config.get(
        "validation_direction_accuracy"
    )

    lgb_weights = ensemble_config.get("lgb_weights")
    baseline_weights = ensemble_config.get("baseline_weights")

    print(
        f"[MODEL READY] version={ensemble_config.get('version')} "
        f"load_time={time.perf_counter() - started:.3f}s "
        f"validation_rmse={ensemble_config.get('validation_rmse')} "
        f"direction_acc={ensemble_config.get('validation_direction_accuracy')}"
    )
    print(f"[MODEL WEIGHTS] lgb={lgb_weights}")
    print(f"[MODEL WEIGHTS] baseline={baseline_weights}")
    print(
        f"[INFERENCE CONFIG] predict_workers={PREDICT_WORKERS} "
        f"lgb_infer_threads={LGB_INFER_THREADS}"
    )


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


def download_history(
    history_url: str,
    target_cols: List[str],
) -> List[Dict[str, Any]]:
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


def history_to_df(
    history: Any,
    history_url: str,
    target_cols: List[str],
) -> pd.DataFrame:
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


def build_predictions(
    pred: np.ndarray,
    target_cols: List[str],
) -> List[Dict[str, Any]]:
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

    pred, timings = predict_future(
        history_df,
        return_timings=True,
    )
    pred = repair_pred(pred, history_df)
    predictions = build_predictions(pred, meta["target_cols"])

    print(
        f"[PREDICT DONE] requestId={meta['request_id']} "
        f"total={timings['total']:.3f}s "
        f"clean={timings['clean']:.3f}s "
        f"feature={timings['feature']:.3f}s "
        f"lgb={timings['lgb']:.3f}s "
        f"xgb={timings['xgb']:.3f}s "
        f"baseline={timings['baseline']:.3f}s "
        f"lgb_outputs={timings['lgb_outputs']} "
        f"lgb_threads={timings['lgb_infer_threads']}"
    )

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
    已通过官网真实接口验证的 callback 格式，固定保持：

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
    request_received_at: float,
) -> None:
    request_id = str(body.get("requestId", "") or "")

    age = time.monotonic() - request_received_at
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

            # 官网明确表示任务已 failed 时，再重试没有意义，会阻塞后续 callback。
            if resp.status_code == 409 and "current status=failed" in response_text:
                print(
                    f"[CALLBACK TERMINAL] requestId={request_id} "
                    f"status=409 response={response_text!r}"
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


def submit_callback(
    callback_url: str,
    body: Dict[str, Any],
    request_received_at: float,
) -> None:
    request_id = str(body.get("requestId", "") or "")
    try:
        CALLBACK_EXECUTOR.submit(
            send_callback,
            callback_url,
            body,
            request_received_at,
        )
        print(
            f"[CALLBACK QUEUED] requestId={request_id} "
            f"serial=true min_age={CALLBACK_MIN_AGE:.2f}s "
            f"gap={CALLBACK_GAP:.2f}s"
        )
    except Exception as exc:
        print(f"[CALLBACK QUEUE FAILED] requestId={request_id}: {exc!r}")


def process_async_request(
    payload: Dict[str, Any],
    callback_url: str,
    callback_token: Any,
    request_received_at: float,
) -> None:
    """后台完成预测，并用已经验证过的 callback JSON 回传结果。"""
    request_id = str(payload.get("requestId", "") or "")
    started = time.perf_counter()

    try:
        result = run_one(payload)
        callback_payload = build_callback_payload(
            request_id=result["requestId"],
            callback_token=callback_token,
            code=0,
            message="success",
            predictions=result["predictions"],
        )
    except Exception as exc:
        print(f"[ASYNC PREDICT ERROR] requestId={request_id}: {exc}")
        traceback.print_exc()
        callback_payload = build_callback_payload(
            request_id=request_id,
            callback_token=callback_token,
            code=-1,
            message=str(exc),
            predictions=[],
        )

    print(
        f"[ASYNC READY] requestId={request_id} "
        f"elapsed={time.perf_counter() - started:.3f}s"
    )
    submit_callback(
        callback_url,
        callback_payload,
        request_received_at,
    )


def submit_async_prediction(
    payload: Dict[str, Any],
    callback_url: str,
    callback_token: Any,
    request_received_at: float,
) -> None:
    request_id = str(payload.get("requestId", "") or "")
    try:
        PREDICT_EXECUTOR.submit(
            process_async_request,
            dict(payload),
            callback_url,
            callback_token,
            request_received_at,
        )
        print(
            f"[ASYNC ACCEPTED] requestId={request_id} "
            f"predict_workers={PREDICT_WORKERS}"
        )
    except Exception as exc:
        print(f"[ASYNC QUEUE FAILED] requestId={request_id}: {exc!r}")
        raise


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
        "version": APP_VERSION,
        "lookback": LOOKBACK,
        "forecast_horizon": HORIZON,
        "target_columns": TARGET_COLUMNS,
        "model_loaded": MODEL_STATUS["loaded"],
        "ensemble_version": MODEL_STATUS["ensemble_version"],
        "validation_rmse": MODEL_STATUS["validation_rmse"],
        "validation_direction_accuracy": MODEL_STATUS[
            "validation_direction_accuracy"
        ],
        "predict_workers": PREDICT_WORKERS,
        "lgb_infer_threads": LGB_INFER_THREADS,
        "callback_timeout": CALLBACK_TIMEOUT,
        "callback_retries": CALLBACK_RETRIES,
        "callback_workers": CALLBACK_WORKERS,
        "callback_serial": True,
        "callback_min_age": CALLBACK_MIN_AGE,
        "callback_gap": CALLBACK_GAP,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any], request: Request):
    request_received_at = time.monotonic()

    auth_error = check_api_key(request)
    if auth_error is not None:
        return auth_error

    try:
        if payload.get("batch") is True:
            return fail("当前联调版本要求 batch_size=1")

        # 这里只做轻量字段校验，绝不在异步请求 HTTP 200 前执行模型预测。
        validate_payload(payload)
        callback_url = str(payload.get("callback_url", "") or "")

        if callback_url:
            submit_async_prediction(
                payload=payload,
                callback_url=callback_url,
                callback_token=payload.get("callback_token"),
                request_received_at=request_received_at,
            )

            # 官方异步规范：接收任务后立即 HTTP 200，计算完成后再 callback。
            return ok(
                {
                    "code": 0,
                    "message": "accepted",
                    "predictions": [],
                }
            )

        # 无 callback_url 时走同步模式，直接返回完整 96 步预测。
        result = run_one(payload)
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
            submit_callback(
                callback_url,
                callback_payload,
                request_received_at,
            )

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
