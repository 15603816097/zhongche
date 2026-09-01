import csv
import json
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

PREDICT_URL = "http://180.127.11.167:10670/predict"
TEST_SEQ = "sequence0001"
FAKE_CALLBACK_PORT = 8888
CALLBACK_TOKEN = "test_token_123"
DATA_PATH = f"./data/raw/{TEST_SEQ}/history.csv"
TARGET_COLUMNS = [
    "vibration_rms", "temperature_c", "current_a",
    "speed_rpm", "acoustic_db", "pressure_kpa"
]
FORECAST_HORIZON = 96
HISTORY_LENGTH = 512

received_callback_payload = None

class MockCallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global received_callback_payload
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length)
        try:
            received_callback_payload = json.loads(body)
            print("\n========✅【模拟平台收到你的callback结果】========")
            print(json.dumps(received_callback_payload, indent=2, ensure_ascii=False))
        except:
            received_callback_payload = None
            print("\n❌ 收到非法JSON")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        return

def start_mock_callback_server():
    server = HTTPServer(("127.0.0.1", FAKE_CALLBACK_PORT), MockCallbackHandler)
    server.serve_forever()

def load_history_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arr = [float(row[c]) for c in TARGET_COLUMNS]
            rows.append(arr)
    return rows

if __name__ == "__main__":
    t = threading.Thread(target=start_mock_callback_server, daemon=True)
    t.start()
    print(f"🔧模拟评测回调服务已启动：127.0.0.1:{FAKE_CALLBACK_PORT}")

    history_data = load_history_csv(DATA_PATH)
    print(f"📄读取history.csv完成，shape={len(history_data)} × {len(history_data[0])}")

    request_body = {
        "requestId": "local-test-seq0001",
        "model": "rail_forecast_model",
        "history_url": "",
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "sampling_interval_seconds": 1,
        "target_columns": TARGET_COLUMNS,
        "history": history_data,
        "callback_url": f"http://127.0.0.1:{FAKE_CALLBACK_PORT}/api/v1/eval/callback",
        "callback_token": CALLBACK_TOKEN
    }

    print("\n🚀向/predict发送测试请求...")
    resp = requests.post(PREDICT_URL, json=request_body, timeout=30)
    print(f"/predict接口返回 status={resp.status_code}")
    print(f"/predict接口返回body: {resp.text[:500]}")

    print("\n⏳等待callback推送结果(最多等待10s)...")
    import time
    for _ in range(10):
        if received_callback_payload is not None:
            break
        time.sleep(1)

    if received_callback_payload is None:
        print("\n❌ 10秒没有收到callback回调！")
    else:
        preds = received_callback_payload.get("predictions")
        if preds is None:
            print("\n❌ callback中缺少 'predictions' 字段！")
            print("收到的完整payload:", json.dumps(received_callback_payload, indent=2))
        else:
            print(f"\n📊 校验 predictions 维度：步数={len(preds)}")
            if len(preds) == FORECAST_HORIZON:
                first = preds[0]
                if "values" in first:
                    vals = first["values"]
                    print(f"每步变量数={len(vals)}")
                    assert len(vals) == len(TARGET_COLUMNS), f"目标列数不对，应该{len(TARGET_COLUMNS)}"
                    print("✅ 维度校验通过！")
                else:
                    print("❌ predictions 元素缺少 'values' 字段")
            else:
                print(f"❌ 预测步数不对，应该{FORECAST_HORIZON}，实际{len(preds)}")
