# test_api.py
import csv
import json
import os
import sys
from typing import Dict, List

import requests

PREDICT_URL = os.getenv("PREDICT_URL", "http://127.0.0.1:8800/predict")
TEST_SEQ = os.getenv("TEST_SEQ", "sequence0001")
DATA_PATH = f"./data/raw/{TEST_SEQ}/history.csv"

TARGET_COLUMNS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

HISTORY_LENGTH = 512
FORECAST_HORIZON = 96


def load_history_csv(csv_path: str) -> List[Dict]:
    history = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for step, row in enumerate(reader):
            history.append(
                {
                    "step": step,
                    "timestamp": row.get("timestamp", ""),
                    "values": {
                        col: float(row[col]) if row[col] not in ("", None) else None
                        for col in TARGET_COLUMNS
                    },
                }
            )

    return history


def validate_response(data: Dict) -> None:
    assert data.get("code") == 0, f"code != 0: {data}"

    predictions = data.get("predictions")
    assert isinstance(predictions, list), "predictions 不是数组"
    assert len(predictions) == FORECAST_HORIZON, (
        f"预测步数错误: {len(predictions)} != {FORECAST_HORIZON}"
    )

    for expected_step, item in enumerate(predictions):
        assert item.get("step") == expected_step, (
            f"step 错误: {item.get('step')} != {expected_step}"
        )

        values = item.get("values")
        assert isinstance(values, dict), f"step={expected_step} 缺少 values"

        assert set(values.keys()) == set(TARGET_COLUMNS), (
            f"step={expected_step} 字段错误: {list(values.keys())}"
        )

        for col in TARGET_COLUMNS:
            value = values[col]
            assert isinstance(value, (int, float)), (
                f"step={expected_step}, {col} 不是数值: {value}"
            )
            assert value == value, f"step={expected_step}, {col} 出现 NaN"
            assert value not in (float("inf"), float("-inf")), (
                f"step={expected_step}, {col} 出现 Inf"
            )


def main():
    print("=" * 70)
    print("Task3 API 单条请求联调")
    print("=" * 70)
    print(f"PREDICT_URL = {PREDICT_URL}")
    print(f"DATA_PATH   = {DATA_PATH}")

    history = load_history_csv(DATA_PATH)
    print(f"history shape = {len(history)} x {len(TARGET_COLUMNS)}")

    if len(history) != HISTORY_LENGTH:
        print(
            f"警告: 官方固定 history_length={HISTORY_LENGTH}，"
            f"当前文件实际为 {len(history)}"
        )

    payload = {
        "requestId": f"local-test-{TEST_SEQ}",
        "model": "rail-timeseries-forecast-v1",
        "history_url": "",
        "history_length": len(history),
        "forecast_horizon": FORECAST_HORIZON,
        "sampling_interval_seconds": 1,
        "target_columns": TARGET_COLUMNS,
        "history": history,
    }

    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": payload["requestId"],
    }

    api_key = os.getenv("API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            PREDICT_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
    except Exception as exc:
        print(f"\n请求失败: {exc}")
        sys.exit(1)

    print(f"\nHTTP status = {response.status_code}")
    print(f"response preview = {response.text[:800]}")

    if response.status_code != 200:
        print("接口没有返回 HTTP 200")
        sys.exit(1)

    try:
        data = response.json()
    except Exception:
        print("响应不是合法 JSON")
        sys.exit(1)

    try:
        validate_response(data)
    except AssertionError as exc:
        print(f"\n校验失败: {exc}")
        sys.exit(1)

    print("\nAPI 校验通过")
    print("code = 0")
    print("predictions = 96")
    print("每步 = 6 个目标字段")
    print("无 NaN / Inf")
    print("\n第一步预测:")
    print(json.dumps(data["predictions"][0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
