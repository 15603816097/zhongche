from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
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

BASE_PORT = int(os.getenv("BASE_PORT", "8800"))
API_PORT = int(os.getenv("API_PORT", "8810"))
CALLBACK_PORT = int(os.getenv("CALLBACK_PORT", "8899"))
N_REQUESTS = int(os.getenv("N_REQUESTS", "50"))
WAIT_TIMEOUT = float(os.getenv("WAIT_TIMEOUT", "600"))

_LOCK = threading.Lock()
_CALLBACKS: dict[str, dict] = {}
_DUPLICATES: list[str] = []
_ERRORS: list[str] = []


def json_value(v):
    if pd.isna(v):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def build_histories():
    out = []
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
        out.append((name, history))
    return out


def sync_predict(port: int, name: str, history: list[dict]) -> np.ndarray:
    payload = {
        "requestId": f"V8_SPEED_SYNC_{name}_{port}",
        "history_length": len(history),
        "forecast_horizon": 96,
        "target_columns": TARGETS,
        "history": history,
    }
    r = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(body)
    pred = body.get("predictions", [])
    if len(pred) != 96:
        raise RuntimeError(f"port={port} prediction count={len(pred)}")
    arr = np.asarray(
        [[row["values"][c] for c in TARGETS] for row in pred],
        dtype=np.float64,
    )
    if arr.shape != (96, 6) or not np.isfinite(arr).all():
        raise RuntimeError(f"port={port} bad prediction shape/values")
    return arr


def validate_callback(body: dict):
    request_id = str(body.get("requestId", ""))
    if not request_id:
        return False, "missing requestId"
    if body.get("callback_token") != f"TOKEN_{request_id}":
        return False, f"{request_id}: token mismatch"
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False, f"{request_id}: invalid results"
    item = results[0]
    if item.get("request_id") != request_id:
        return False, f"{request_id}: nested request_id mismatch"
    data = item.get("data")
    if not isinstance(data, dict):
        return False, f"{request_id}: missing data"
    if data.get("code") != 0 or data.get("message") != "success":
        return False, f"{request_id}: bad code/message"
    pred = data.get("predictions")
    if not isinstance(pred, list) or len(pred) != 96:
        return False, f"{request_id}: bad predictions"
    return True, request_id


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            ok, info = validate_callback(body)
            with _LOCK:
                if not ok:
                    _ERRORS.append(info)
                elif info in _CALLBACKS:
                    _DUPLICATES.append(info)
                else:
                    _CALLBACKS[info] = body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as exc:
            with _LOCK:
                _ERRORS.append(repr(exc))
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        return


def main() -> int:
    histories = build_histories()

    base_health = requests.get(
        f"http://127.0.0.1:{BASE_PORT}/health", timeout=10
    ).json()
    cand_health = requests.get(
        f"http://127.0.0.1:{API_PORT}/health", timeout=10
    ).json()
    print("base health:", base_health)
    print("candidate health:", cand_health)

    if base_health.get("version") != "2.9.0":
        raise SystemExit("baseline 8800 is not V8")
    if cand_health.get("version") != "2.9.0":
        raise SystemExit("candidate is not V8")

    print("\n[1/2] Exact prediction reproduction")
    max_diff = 0.0
    for name, history in histories:
        base = sync_predict(BASE_PORT, name, history)
        cand = sync_predict(API_PORT, name, history)
        diff = float(np.max(np.abs(base - cand)))
        max_diff = max(max_diff, diff)
        print(f"{name}: max_abs_diff={diff:.3e}")

    print(f"overall max_abs_diff={max_diff:.3e}")
    if max_diff > 1e-7:
        raise SystemExit(f"FAIL: prediction drift={max_diff}")

    print("\n[2/2] 50-request callback stress")
    server = ThreadingHTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    accepted = 0
    submit_errors: list[str] = []
    accept_latencies: list[float] = []
    started = time.perf_counter()

    try:
        for i in range(N_REQUESTS):
            name, history = histories[i % len(histories)]
            rid = f"V8_SPEED_{i:03d}_{name}"
            payload = {
                "requestId": rid,
                "history_length": len(history),
                "forecast_horizon": 96,
                "target_columns": TARGETS,
                "history": history,
                "callback_url": f"http://127.0.0.1:{CALLBACK_PORT}/callback",
                "callback_token": f"TOKEN_{rid}",
            }
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    f"http://127.0.0.1:{API_PORT}/predict",
                    json=payload,
                    timeout=20,
                )
                accept_latencies.append(time.perf_counter() - t0)
                b = r.json()
                if (
                    r.status_code == 200
                    and b.get("code") == 0
                    and b.get("message") == "accepted"
                    and b.get("predictions") == []
                ):
                    accepted += 1
                else:
                    submit_errors.append(f"{rid}: HTTP={r.status_code} body={b}")
            except Exception as exc:
                submit_errors.append(f"{rid}: {exc!r}")

        submit_seconds = time.perf_counter() - started
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            with _LOCK:
                count = len(_CALLBACKS)
            if count >= N_REQUESTS:
                break
            time.sleep(0.25)

        total_seconds = time.perf_counter() - started
        with _LOCK:
            callbacks = len(_CALLBACKS)
            duplicates = len(_DUPLICATES)
            errors = len(_ERRORS)

        mean_accept = (
            sum(accept_latencies) / len(accept_latencies)
            if accept_latencies
            else float("nan")
        )
        max_accept = max(accept_latencies) if accept_latencies else float("nan")
        throughput = callbacks / total_seconds if total_seconds > 0 else 0.0

        print("\n" + "=" * 100)
        print("V8-SPEED CANDIDATE SUMMARY")
        print("=" * 100)
        print(f"accepted          : {accepted}/{N_REQUESTS}")
        print(f"callbacks         : {callbacks}/{N_REQUESTS}")
        print(f"duplicates        : {duplicates}")
        print(f"callback errors   : {errors}")
        print(f"submit errors     : {len(submit_errors)}")
        print(f"submit seconds    : {submit_seconds:.3f}")
        print(f"mean accept       : {mean_accept:.4f}s")
        print(f"max accept        : {max_accept:.4f}s")
        print(f"total seconds     : {total_seconds:.3f}")
        print(f"throughput        : {throughput:.3f} req/s")
        print(f"prediction diff   : {max_diff:.3e}")

        passed = (
            accepted == N_REQUESTS
            and callbacks == N_REQUESTS
            and duplicates == 0
            and errors == 0
            and not submit_errors
            and max_diff <= 1e-7
        )

        print(
            "RESULT "
            f"pass={int(passed)} "
            f"max_diff={max_diff:.12e} "
            f"total_seconds={total_seconds:.6f} "
            f"throughput={throughput:.6f} "
            f"accepted={accepted} "
            f"callbacks={callbacks} "
            f"duplicates={duplicates} "
            f"errors={errors + len(submit_errors)}"
        )
        print(f"V8-SPEED TEST: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
