from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import requests

TARGETS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

API_URL = "http://127.0.0.1:8801/predict"
HEALTH_URL = "http://127.0.0.1:8801/health"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8899
CALLBACK_URL = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"
N_REQUESTS = 50
WAIT_TIMEOUT = 600.0

_LOCK = threading.Lock()
_CALLBACKS: dict[str, dict] = {}
_DUPLICATES: list[str] = []
_CALLBACK_ERRORS: list[str] = []


def json_value(v):
    if pd.isna(v):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def build_histories():
    histories = []
    for seq_num in range(1, 6):
        name = f"sequence{seq_num:04d}"
        df = pd.read_csv(f"data/raw/{name}/history.csv")
        history = []
        for i, (_, row) in enumerate(df.iterrows()):
            history.append(
                {
                    "step": i,
                    "values": {c: json_value(row[c]) for c in TARGETS},
                }
            )
        histories.append((name, history))
    return histories


def validate_callback(body: dict) -> tuple[bool, str]:
    try:
        request_id = str(body.get("requestId", ""))
        if not request_id:
            return False, "missing top-level requestId"
        if body.get("callback_token") != f"TOKEN_{request_id}":
            return False, f"{request_id}: callback_token mismatch"

        results = body.get("results")
        if not isinstance(results, list) or len(results) != 1:
            return False, f"{request_id}: invalid results"
        item = results[0]
        if item.get("request_id") != request_id:
            return False, f"{request_id}: nested request_id mismatch"
        data = item.get("data")
        if not isinstance(data, dict):
            return False, f"{request_id}: missing nested data"
        if data.get("code") != 0 or data.get("message") != "success":
            return False, f"{request_id}: callback code/message={data.get('code')}/{data.get('message')}"

        predictions = data.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 96:
            return False, f"{request_id}: prediction count={len(predictions) if isinstance(predictions, list) else 'invalid'}"
        for step, p in enumerate(predictions):
            if p.get("step") != step:
                return False, f"{request_id}: bad step at {step}"
            values = p.get("values")
            if not isinstance(values, dict) or set(values) != set(TARGETS):
                return False, f"{request_id}: target fields invalid at step={step}"
            for c in TARGETS:
                v = values[c]
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    return False, f"{request_id}: nonfinite {c} at step={step}"
        return True, request_id
    except Exception as exc:
        return False, repr(exc)


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            body = json.loads(raw.decode("utf-8"))
            ok, info = validate_callback(body)
            with _LOCK:
                if not ok:
                    _CALLBACK_ERRORS.append(info)
                else:
                    rid = info
                    if rid in _CALLBACKS:
                        _DUPLICATES.append(rid)
                    else:
                        _CALLBACKS[rid] = body
                    count = len(_CALLBACKS)
                    if count % 5 == 0 or count == N_REQUESTS:
                        print(f"callback progress: {count}/{N_REQUESTS}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as exc:
            with _LOCK:
                _CALLBACK_ERRORS.append(f"receiver error: {exc!r}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        return


def main() -> int:
    health = requests.get(HEALTH_URL, timeout=10).json()
    print("V9-Safe health:", health)
    if health.get("status") != "ok" or health.get("version") != "2.9.0-v9-safe":
        raise SystemExit("FAIL: V9-Safe 8801 not healthy")

    histories = build_histories()
    server = ThreadingHTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"local callback receiver: {CALLBACK_URL}")

    accepted = 0
    submit_errors = []
    accepted_latencies = []
    started = time.perf_counter()

    try:
        for i in range(N_REQUESTS):
            seq_name, history = histories[i % len(histories)]
            request_id = f"V9_STRESS_{i:03d}"
            payload = {
                "requestId": request_id,
                "history_length": len(history),
                "forecast_horizon": 96,
                "target_columns": TARGETS,
                "history": history,
                "callback_url": CALLBACK_URL,
                "callback_token": f"TOKEN_{request_id}",
            }
            t0 = time.perf_counter()
            try:
                resp = requests.post(API_URL, json=payload, timeout=20)
                dt = time.perf_counter() - t0
                accepted_latencies.append(dt)
                data = resp.json()
                if resp.status_code == 200 and data.get("code") == 0 and data.get("message") == "accepted" and data.get("predictions") == []:
                    accepted += 1
                else:
                    submit_errors.append(
                        f"{request_id}: HTTP={resp.status_code} body={data}"
                    )
            except Exception as exc:
                submit_errors.append(f"{request_id}: {exc!r}")

        submit_elapsed = time.perf_counter() - started
        print(f"accepted: {accepted}/{N_REQUESTS}")
        print(f"submit elapsed: {submit_elapsed:.3f}s")
        if accepted_latencies:
            print(f"mean accept latency: {sum(accepted_latencies)/len(accepted_latencies):.4f}s")
            print(f"max accept latency : {max(accepted_latencies):.4f}s")

        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            with _LOCK:
                count = len(_CALLBACKS)
            if count >= N_REQUESTS:
                break
            time.sleep(0.5)

        total_elapsed = time.perf_counter() - started
        with _LOCK:
            callback_count = len(_CALLBACKS)
            duplicates = list(_DUPLICATES)
            callback_errors = list(_CALLBACK_ERRORS)

        print("\n" + "=" * 100)
        print("V9-SAFE 50 REQUEST CALLBACK STRESS SUMMARY")
        print("=" * 100)
        print(f"accepted requests : {accepted}/{N_REQUESTS}")
        print(f"callbacks received: {callback_count}/{N_REQUESTS}")
        print(f"duplicate callbacks: {len(duplicates)}")
        print(f"callback errors    : {len(callback_errors)}")
        print(f"submit errors      : {len(submit_errors)}")
        print(f"total elapsed      : {total_elapsed:.3f}s")

        if submit_errors:
            print("submit error sample:", submit_errors[:3])
        if callback_errors:
            print("callback error sample:", callback_errors[:3])
        if duplicates:
            print("duplicate sample:", duplicates[:3])

        health_after = requests.get(HEALTH_URL, timeout=10).json()
        print("health after       :", health_after.get("status"), health_after.get("version"))

        passed = (
            accepted == N_REQUESTS
            and callback_count == N_REQUESTS
            and not duplicates
            and not callback_errors
            and not submit_errors
            and health_after.get("status") == "ok"
            and health_after.get("version") == "2.9.0-v9-safe"
        )
        print(f"\n50-REQUEST CALLBACK STRESS TEST: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
