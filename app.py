# app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import requests
import json
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.inference import predict_future
from config import TARGET_COLUMNS

app = FastAPI(title="轨道交通时序预测API")


# ========== 请求模型 ==========
class HistoryItem(BaseModel):
    step: int
    timestamp: str
    values: Dict[str, float]

class PredictRequest(BaseModel):
    requestId: str
    model: str
    history_url: str
    history_length: int
    forecast_horizon: int
    sampling_interval_seconds: int
    target_columns: List[str]
    history: List[Dict[str, Any]]  # 可以是多种格式
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None


# ========== 核心推理函数 ==========
def run_inference(history_data: List[Dict], forecast_horizon: int, target_cols: List[str]) -> List[Dict]:
    """
    执行推理，返回符合官方格式的 predictions 数组
    """
    # 1. 解析 history 数据
    rows = []
    for item in history_data:
        # 格式1：{"step": 0, "timestamp": "...", "values": {"vibration_rms": 1.2, ...}}
        if isinstance(item, dict):
            if "values" in item and isinstance(item["values"], dict):
                vals = item["values"]
                row = {}
                for col in target_cols:
                    val = vals.get(col, 0.0)
                    if val is None:
                        val = 0.0
                    row[col] = float(val)
                rows.append(row)
            elif "step" in item and "timestamp" in item:
                # 格式2：{"step": 0, "timestamp": "...", "vibration_rms": 1.2, ...}
                row = {}
                for col in target_cols:
                    val = item.get(col, 0.0)
                    if val is None:
                        val = 0.0
                    row[col] = float(val)
                rows.append(row)
            else:
                raise ValueError(f"未知的 history 格式: {item}")
        elif isinstance(item, list):
            # 格式3：[v1, v2, v3, ...]
            row = {}
            for i, col in enumerate(target_cols):
                val = item[i] if i < len(item) else 0.0
                if val is None:
                    val = 0.0
                row[col] = float(val)
            rows.append(row)
        else:
            raise ValueError(f"未知的 history 类型: {type(item)}")

    history_df = pd.DataFrame(rows)

    # 按 step 排序（如果有 step 列）
    if "step" in history_df.columns:
        history_df = history_df.sort_values("step")

    print(f"📊 转换后 DataFrame shape: {history_df.shape}")

    # 2. 调用真正的推理函数
    pred_array = predict_future(history_df)  # shape: (96, 6)

    # 3. 修复非法数值（NaN, Inf）
    pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=9999.0, neginf=-9999.0)

    # 4. 构建预测结果的索引映射
    col_index = {name: i for i, name in enumerate(TARGET_COLUMNS)}
    ordered_indices = [col_index[col] for col in target_cols if col in col_index]
    if not ordered_indices:
        ordered_indices = list(range(len(target_cols)))

    # 5. 生成 predictions 数组（官方格式）
    predictions = []
    for step in range(forecast_horizon):
        step_values = {}
        for idx, col in enumerate(target_cols):
            if idx < len(ordered_indices) and ordered_indices[idx] < len(pred_array[step]):
                step_values[col] = float(pred_array[step][ordered_indices[idx]])
            else:
                step_values[col] = 0.0
        predictions.append({
            "step": step,
            "values": step_values
        })

    return predictions


# ========== 核心 API 端点 ==========
@app.post("/predict")
async def predict(req: PredictRequest):
    print(f"📨 收到请求，字段: {list(req.model_dump().keys())}")
    print(f"📊 收到 {len(req.history)} 步历史数据")

    try:
        # 1. 执行推理
        predictions = run_inference(
            history_data=req.history,
            forecast_horizon=req.forecast_horizon,
            target_cols=req.target_columns
        )

        # 2. 构造响应体（官方格式）
        response_body = {
            "code": 0,
            "message": "success",
            "predictions": predictions
        }

        # 3. 如果有回调地址，异步发送回调
        if req.callback_url:
            # 回调 payload：必须包含 callback_token 和业务结果
            callback_payload = {
                "callback_token": req.callback_token,
                "code": 0,
                "message": "success",
                "predictions": predictions
            }
            print(f"📤 发送回调到: {req.callback_url}")

            try:
                # 添加必要的请求头
                headers = {"Content-Type": "application/json"}
                # 如果有 Authorization token（平台可能要求）
                # if req.callback_token:
                #     headers["Authorization"] = f"Bearer {req.callback_token}"

                resp = requests.post(
                    req.callback_url,
                    json=callback_payload,
                    timeout=12,
                    headers=headers
                )
                print(f"✅ 回调响应: status={resp.status_code}, body={resp.text[:300]}")
            except requests.exceptions.Timeout:
                print("❌ 回调超时")
            except requests.exceptions.ConnectionError:
                print("❌ 回调连接失败（可能是本地测试环境）")
            except Exception as e:
                print(f"❌ 回调异常: {str(e)}")

        # 4. 同步返回结果（平台优先读取同步响应）
        return JSONResponse(content=response_body)

    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        traceback.print_exc()
        # 返回错误格式（code 非 0）
        return JSONResponse(
            status_code=200,
            content={
                "code": -1,
                "message": str(e),
                "predictions": []
            }
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
