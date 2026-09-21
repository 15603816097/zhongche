from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

API_PORT = int(os.getenv("API_PORT", "8822"))
CALLBACK_PORT = int(os.getenv("CALLBACK_PORT", "8896"))
N_REQUESTS = int(os.getenv("N_REQUESTS", "50"))
WAIT_TIMEOUT = float(os.getenv("WAIT_TIMEOUT", "600"))
EXPECTED_PATH = Path(
    os.getenv(
        "EXPECTED_PATH",
        "artifacts/fa52_acoustic075/expected_predictions.npz",
    )
)

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
        history = [
            {
                "step": i,
                "values": {c: json_value(row[c]) for c in TARGETS},
            }
            for i, (_, row) in enumerate(df.iterrows())
        ]
        out.append((name, history))
    return out


def sync_predict(name, history):
    payload = {
        "requestId": f"FA52_A075_SYNC_{name}",
        "history_length": len(history),
        "forecast_horizon": 96,
        "target_columns": TARGETS,
        "history": history,
    }
    r = requests.post(
        f"http://127.0.0.1:{API_PORT}/predict",
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(body)
    arr = np.asarray(
        [
            [row["values"][c] for c in TARGETS]
            for row in body.get("predictions", [])
        ],
        dtype=np.float64,
    )
    if arr.shape != (96, 6) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad prediction shape={arr.shape}")
    return arr


def validate_callback(body):
    rid = str(body.get("requestId", ""))
    if not rid:
        return False, "missing requestId"
    if body.get("callback_token") != f"TOKEN_{rid}":
        return False, f"{rid}: token mismatch"
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False, f"{rid}: invalid results"
    item = results[0]
    if item.get("request_id") != rid:
        return False, f"{rid}: nested request_id mismatch"
    data = item.get("data")
    if not isinstance(data, dict):
        return False, f"{rid}: missing data"
    if data.get("code") != 0 or data.get("message") != "success":
        return False, f"{rid}: bad code/message"
    pred = data.get("predictions")
    if not isinstance(pred, list) or len(pred) != 96:
        return False, f"{rid}: bad predictions"
    return True, rid


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
    if not EXPECTED_PATH.is_file():
        raise SystemExit(f"missing expected artifact: {EXPECTED_PATH}")

    z = np.load(EXPECTED_PATH, allow_pickle=False)
    names = [str(x) for x in z["names"].tolist()]
    expected = np.asarray(z["expected"], dtype=np.float64)
    histories = build_histories()

    if names != [name for name, _ in histories]:
        raise SystemExit(f"name mismatch: {names}")

    health = requests.get(
        f"http://127.0.0.1:{API_PORT}/health",
        timeout=10,
    ).json()
    info = requests.get(
        f"http://127.0.0.1:{API_PORT}/candidate",
        timeout=10,
    ).json()
    print("health:", health)
    print("candidate:", info)

    if health.get("version") != "3.0.1-fa52-acoustic075":
        raise SystemExit("candidate version mismatch")
    if info.get("shrink_scales", {}).get("acoustic_db") != 0.75:
        raise SystemExit("candidate scale mismatch")

    print("\n[1/2] Exact online reproduction")
    max_diff = 0.0
    for i, (name, history) in enumerate(histories):
        pred = sync_predict(name, history)
        diff = float(np.max(np.abs(pred - expected[i])))
        max_diff = max(max_diff, diff)
        print(f"{name}: max_abs_diff={diff:.3e}")

    print(f"overall max_abs_diff={max_diff:.3e}")
    if max_diff > 1e-10:
        raise SystemExit(f"FAIL: formula drift={max_diff}")

    print("\n[2/2] 50-request callback stress")
    server = ThreadingHTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    accepted = 0
    submit_errors = []
    accept_latencies = []
    started = time.perf_counter()

    try:
        for i in range(N_REQUESTS):
            name, history = histories[i % len(histories)]
            rid = f"FA52_A075_{i:03d}_{name}"
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
                    submit_errors.append(
                        f"{rid}: HTTP={r.status_code} body={b}"
                    )
            except Exception as exc:
                submit_errors.append(f"{rid}: {exc!r}")

        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            with _LOCK:
                count = len(_CALLBACKS)
            if count >= N_REQUESTS:
                break
            time.sleep(0.25)

        total = time.perf_counter() - started
        with _LOCK:
            callbacks = len(_CALLBACKS)
            duplicates = len(_DUPLICATES)
            errors = len(_ERRORS)

        throughput = callbacks / total if total > 0 else 0.0
        mean_accept = (
            sum(accept_latencies) / len(accept_latencies)
            if accept_latencies
            else float("nan")
        )

        print("\n" + "=" * 100)
        print("FA52 ACOUSTIC075 ONLINE SUMMARY")
        print("=" * 100)
        print(f"accepted          : {accepted}/{N_REQUESTS}")
        print(f"callbacks         : {callbacks}/{N_REQUESTS}")
        print(f"duplicates        : {duplicates}")
        print(f"callback errors   : {errors}")
        print(f"submit errors     : {len(submit_errors)}")
        print(f"total seconds     : {total:.3f}")
        print(f"throughput        : {throughput:.3f} req/s")
        print(f"mean accept       : {mean_accept:.4f}s")
        print(f"prediction diff   : {max_diff:.3e}")

        passed = (
            accepted == N_REQUESTS
            and callbacks == N_REQUESTS
            and duplicates == 0
            and errors == 0
            and not submit_errors
            and max_diff <= 1e-10
        )
        print(f"FA52 ACOUSTIC075 ONLINE GATE: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
