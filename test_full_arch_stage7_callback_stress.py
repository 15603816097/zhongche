from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, HORIZON, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parent
STAGE5_NPZ = ROOT / "artifacts" / "full_arch" / "dynamic_gate" / "stage5_predictions.npz"
API_URL = "http://127.0.0.1:8810"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8811
CALLBACK_URL = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"
N_REQUESTS = 50
TIMEOUT_SECONDS = 240.0

_lock = threading.Lock()
_callbacks: dict[str, dict] = {}
_callback_errors: list[str] = []
_callback_order: list[str] = []


def json_safe_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


class CallbackHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))

            token = body.get("callback_token")
            results = body.get("results")
            if not isinstance(results, list) or len(results) != 1:
                raise ValueError("results must be a single-item list")
            row = results[0]
            request_id = str(row.get("request_id", ""))
            data = row.get("data")
            if not request_id:
                raise ValueError("missing nested request_id")
            if not isinstance(data, dict):
                raise ValueError("missing nested data")
            if int(data.get("code", -999)) != 0:
                raise ValueError(
                    f"callback code={data.get('code')} message={data.get('message')}"
                )
            predictions = data.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != HORIZON:
                raise ValueError(
                    f"{request_id}: prediction count={0 if not isinstance(predictions, list) else len(predictions)}"
                )
            for step, pred in enumerate(predictions):
                if int(pred.get("step", -1)) != step:
                    raise ValueError(f"{request_id}: bad step at {step}")
                values = pred.get("values")
                if not isinstance(values, dict):
                    raise ValueError(f"{request_id}: missing values at {step}")
                if set(values) != set(TARGET_COLUMNS):
                    raise ValueError(f"{request_id}: target columns mismatch at {step}")
                for col in TARGET_COLUMNS:
                    value = float(values[col])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"{request_id}: non-finite {col} at step={step}"
                        )

            with _lock:
                if request_id in _callbacks:
                    _callback_errors.append(f"duplicate callback: {request_id}")
                else:
                    _callbacks[request_id] = {
                        "token": token,
                        "body": body,
                        "received_at": time.perf_counter(),
                    }
                    _callback_order.append(request_id)

            payload = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            with _lock:
                _callback_errors.append(repr(exc))
            payload = b'{"ok":false}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)


def build_payload(seq_num: int, request_id: str) -> dict:
    name = f"sequence{seq_num:04d}"
    hdf = pd.read_csv(DATA_DIR / name / "history.csv")
    history = [
        {
            "step": int(i),
            "values": {
                col: json_safe_float(row[col])
                for col in TARGET_COLUMNS
            },
        }
        for i, row in hdf.iterrows()
    ]
    return {
        "requestId": request_id,
        "history_length": len(history),
        "forecast_horizon": HORIZON,
        "target_columns": list(TARGET_COLUMNS),
        "history": history,
        "callback_url": CALLBACK_URL,
        "callback_token": f"TOKEN_{request_id}",
    }


def callback_array(body: dict) -> np.ndarray:
    predictions = body["results"][0]["data"]["predictions"]
    return np.asarray(
        [[row["values"][col] for col in TARGET_COLUMNS] for row in predictions],
        dtype=np.float64,
    )


def main() -> int:
    if not STAGE5_NPZ.is_file():
        raise FileNotFoundError(STAGE5_NPZ)

    with np.load(STAGE5_NPZ) as z:
        expected = np.asarray(z["final_known_prediction"], dtype=np.float64)

    health = requests.get(f"{API_URL}/health", timeout=20)
    health.raise_for_status()
    h = health.json()
    if h.get("version") != "3.0.0-full-arch":
        raise RuntimeError(f"unexpected API version: {h.get('version')}")
    if not h.get("model_loaded"):
        raise RuntimeError("health says model_loaded=false")

    server = ThreadingHTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("=" * 118)
    print("FULL ARCHITECTURE - STAGE 7 API + 50 CALLBACK STRESS")
    print("=" * 118)
    print("api              :", API_URL)
    print("api version      :", h.get("version"))
    print("ensemble version :", h.get("ensemble_version"))
    print("predict workers  :", h.get("predict_workers"))
    print("lgb threads      :", h.get("lgb_infer_threads"))
    print("callback serial  :", h.get("callback_serial"))
    print("callback min age :", h.get("callback_min_age"))
    print("callback gap     :", h.get("callback_gap"))
    print("requests         :", N_REQUESTS)

    request_to_seq = {}
    accept_times = []
    submitted_at = time.perf_counter()

    try:
        for i in range(N_REQUESTS):
            seq_num = (i % 5) + 1
            request_id = f"FULL_ARCH_STRESS_{i+1:03d}_SEQ_{seq_num:04d}"
            request_to_seq[request_id] = seq_num
            payload = build_payload(seq_num, request_id)

            started = time.perf_counter()
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            elapsed = time.perf_counter() - started
            accept_times.append(elapsed)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"{request_id}: HTTP {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()
            if data.get("code") != 0 or data.get("message") != "accepted":
                raise RuntimeError(f"{request_id}: bad accept response {data}")
            if data.get("predictions") != []:
                raise RuntimeError(f"{request_id}: async accept returned predictions")

        submit_seconds = time.perf_counter() - submitted_at
        print(
            f"submitted 50/50    : {submit_seconds:.3f}s "
            f"(mean_accept={np.mean(accept_times):.4f}s "
            f"max_accept={np.max(accept_times):.4f}s)"
        )

        deadline = time.monotonic() + TIMEOUT_SECONDS
        last_print = -1
        while time.monotonic() < deadline:
            with _lock:
                count = len(_callbacks)
                errors = list(_callback_errors)
            if errors:
                raise RuntimeError(f"callback receiver errors: {errors[:5]}")
            if count == N_REQUESTS:
                break
            bucket = count // 10
            if bucket != last_print:
                print(f"callbacks received : {count}/{N_REQUESTS}", flush=True)
                last_print = bucket
            time.sleep(0.25)

        total_seconds = time.perf_counter() - submitted_at
        with _lock:
            callbacks = dict(_callbacks)
            errors = list(_callback_errors)

        if errors:
            raise RuntimeError(f"callback receiver errors: {errors}")
        if len(callbacks) != N_REQUESTS:
            missing = sorted(set(request_to_seq) - set(callbacks))
            raise RuntimeError(
                f"callback timeout: received={len(callbacks)}/{N_REQUESTS} "
                f"missing={missing[:10]}"
            )

        max_diff = 0.0
        bad_tokens = 0
        for request_id, seq_num in request_to_seq.items():
            item = callbacks[request_id]
            expected_token = f"TOKEN_{request_id}"
            if item["token"] != expected_token:
                bad_tokens += 1
            arr = callback_array(item["body"])
            diff = float(np.max(np.abs(arr - expected[seq_num - 1])))
            max_diff = max(max_diff, diff)

        if bad_tokens:
            raise RuntimeError(f"callback token mismatches: {bad_tokens}")
        if max_diff > 2e-5:
            raise RuntimeError(
                f"callback prediction reproduction mismatch: max_diff={max_diff}"
            )

        print("\n" + "=" * 118)
        print("STAGE 7 STRESS SUMMARY")
        print("=" * 118)
        print(f"accepted            : {N_REQUESTS}/{N_REQUESTS}")
        print(f"callbacks           : {len(callbacks)}/{N_REQUESTS}")
        print(f"duplicates/errors   : 0/0")
        print(f"callback token      : PASS")
        print(f"prediction max diff : {max_diff:.3e}")
        print(f"submit seconds      : {submit_seconds:.3f}")
        print(f"total seconds       : {total_seconds:.3f}")
        print(f"mean throughput     : {N_REQUESTS / max(total_seconds, 1e-9):.3f} req/s")
        print("STAGE 7 CALLBACK STRESS: PASS")
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
