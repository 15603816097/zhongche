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

from src.data_cleaner import clean_sequence
from src.trajectory_fusion import endpoint_zero_highpass
from v9_analog_multiscale_diagnostic import evaluate_rows, proxy_gain

TARGETS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

BASE_PORT = int(os.getenv("BASE_PORT", "8800"))
API_PORT = int(os.getenv("API_PORT", "8811"))
CALLBACK_PORT = int(os.getenv("CALLBACK_PORT", "8898"))
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


def histories():
    out = []
    for seq_num in range(1, 6):
        name = f"sequence{seq_num:04d}"
        hdf = pd.read_csv(f"data/raw/{name}/history.csv")
        fdf = pd.read_csv(f"data/raw/{name}/future.csv")
        history = [
            {
                "step": i,
                "values": {c: json_value(row[c]) for c in TARGETS},
            }
            for i, (_, row) in enumerate(hdf.iterrows())
        ]
        truth = fdf[TARGETS].iloc[:96].to_numpy(dtype=np.float64)
        clean = clean_sequence(hdf[TARGETS])
        anchor = clean[TARGETS].iloc[-1].to_numpy(dtype=np.float64)
        out.append((name, history, truth, anchor))
    return out


def call_sync(port, name, history):
    payload = {
        "requestId": f"MSF_SYNC_{name}_{port}",
        "history_length": len(history),
        "forecast_horizon": 96,
        "target_columns": TARGETS,
        "history": history,
    }
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/predict", json=payload, timeout=180)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(body)
    pred = body.get("predictions", [])
    arr = np.asarray(
        [[row["values"][c] for c in TARGETS] for row in pred],
        dtype=np.float64,
    )
    if arr.shape != (96, 6) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad prediction from port={port}")
    return arr, dt


def expected_msf(base, config):
    out = base.copy()
    for name in config["enabled_targets"]:
        j = TARGETS.index(name)
        gains = config["selected"][name]["gains"]
        x = base[:, j]
        hp5 = endpoint_zero_highpass(x, 5)
        hp13 = endpoint_zero_highpass(x, 13)
        hp33 = endpoint_zero_highpass(x, 33)
        short = hp5
        medium = hp13 - hp5
        long_local = hp33 - hp13
        out[:, j] = (
            x
            + float(gains[0]) * short
            + float(gains[1]) * medium
            + float(gains[2]) * long_local
        )
    return out


def validate_callback(body):
    rid = str(body.get("requestId", ""))
    if not rid:
        return False, "missing requestId"
    if body.get("callback_token") != f"TOKEN_{rid}":
        return False, f"{rid}: token mismatch"
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False, f"{rid}: bad results"
    item = results[0]
    if item.get("request_id") != rid:
        return False, f"{rid}: nested id mismatch"
    data = item.get("data")
    if not isinstance(data, dict) or data.get("code") != 0:
        return False, f"{rid}: bad nested data"
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

    def log_message(self, fmt, *args):
        return


def main():
    config = json.load(open("models/v8_multiscale_fusion_candidate.json", encoding="utf-8"))
    if not config.get("offline_gate_pass"):
        raise SystemExit("offline gate did not pass")

    bh = requests.get(f"http://127.0.0.1:{BASE_PORT}/health", timeout=10).json()
    ch = requests.get(f"http://127.0.0.1:{API_PORT}/health", timeout=10).json()
    print("base health:", bh)
    print("candidate health:", ch)
    if ch.get("version") != "2.9.0-v8-msf":
        raise SystemExit("candidate version mismatch")

    rows = histories()
    truths, base_preds, cand_preds, anchors = [], [], [], []
    base_times, cand_times = [], []
    max_formula_diff = 0.0

    print("\n[1/2] Online formula reproduction + 5-sequence metrics")
    for name, history, truth, anchor in rows:
        b, tb = call_sync(BASE_PORT, name, history)
        c, tc = call_sync(API_PORT, name, history)
        expected = expected_msf(b, config)
        diff = float(np.max(np.abs(c - expected)))
        max_formula_diff = max(max_formula_diff, diff)
        print(
            f"{name} base={tb:.3f}s msf={tc:.3f}s "
            f"extra={tc-tb:+.3f}s formula_diff={diff:.3e}"
        )
        truths.append(truth)
        base_preds.append(b)
        cand_preds.append(c)
        anchors.append(anchor)
        base_times.append(tb)
        cand_times.append(tc)

    truth = np.stack(truths)
    base = np.stack(base_preds)
    cand = np.stack(cand_preds)
    anchors = np.stack(anchors)

    base_rmse = float(np.sqrt(np.mean((truth-base)**2)))
    cand_rmse = float(np.sqrt(np.mean((truth-cand)**2)))
    proxy_base, proxy_cand, trend_base, trend_cand = [], [], [], []
    for j, name in enumerate(TARGETS):
        b = evaluate_rows(truth[:, :, j], base[:, :, j], anchors[:, j])
        c = evaluate_rows(truth[:, :, j], cand[:, :, j], anchors[:, j])
        proxy_base.append(b["proxy_loss"])
        proxy_cand.append(c["proxy_loss"])
        trend_base.append(b["trend_core"])
        trend_cand.append(c["trend_core"])
        print(
            f"{name:16s} rmse_ratio={c['rmse']/max(b['rmse'],1e-12):.6f} "
            f"proxy_gain={100*proxy_gain(b,c):+.2f}% "
            f"trend_gain={c['trend_core']-b['trend_core']:+.5f}"
        )

    mean_proxy_gain = (
        (np.mean(proxy_base)-np.mean(proxy_cand))
        / max(abs(np.mean(proxy_base)),1e-12)
    )
    mean_trend_gain = float(np.mean(trend_cand)-np.mean(trend_base))

    print("\nmetric summary:")
    print(f"flat RMSE base/cand : {base_rmse:.6f} / {cand_rmse:.6f}")
    print(f"flat RMSE ratio     : {cand_rmse/max(base_rmse,1e-12):.6f}")
    print(f"mean proxy gain     : {100*mean_proxy_gain:+.2f}%")
    print(f"mean trend gain     : {mean_trend_gain:+.6f}")
    print(f"formula max diff    : {max_formula_diff:.3e}")
    print(f"mean sync base time : {np.mean(base_times):.3f}s")
    print(f"mean sync MSF time  : {np.mean(cand_times):.3f}s")

    if max_formula_diff > 1e-8:
        raise SystemExit("FAIL: online MSF formula mismatch")
    if cand_rmse >= base_rmse:
        raise SystemExit("FAIL: MSF RMSE did not improve")
    if mean_proxy_gain <= 0:
        raise SystemExit("FAIL: MSF proxy did not improve")

    print("\n[2/2] 50-request callback stress")
    server = ThreadingHTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    accepted = 0
    submit_errors = []
    accept_latencies = []
    started = time.perf_counter()
    try:
        for i in range(N_REQUESTS):
            name, history, _, _ = rows[i % len(rows)]
            rid = f"MSF_STRESS_{i:03d}_{name}"
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
                accept_latencies.append(time.perf_counter()-t0)
                body = r.json()
                if (
                    r.status_code == 200
                    and body.get("code") == 0
                    and body.get("message") == "accepted"
                    and body.get("predictions") == []
                ):
                    accepted += 1
                else:
                    submit_errors.append(f"{rid}: HTTP={r.status_code} body={body}")
            except Exception as exc:
                submit_errors.append(f"{rid}: {exc!r}")

        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            with _LOCK:
                count = len(_CALLBACKS)
            if count >= N_REQUESTS:
                break
            time.sleep(0.25)

        total = time.perf_counter()-started
        with _LOCK:
            callbacks = len(_CALLBACKS)
            duplicates = len(_DUPLICATES)
            errors = len(_ERRORS)

        print("\n" + "="*100)
        print("V8-MSF ONLINE VALIDATION SUMMARY")
        print("="*100)
        print(f"accepted          : {accepted}/{N_REQUESTS}")
        print(f"callbacks         : {callbacks}/{N_REQUESTS}")
        print(f"duplicates        : {duplicates}")
        print(f"callback errors   : {errors}")
        print(f"submit errors     : {len(submit_errors)}")
        print(f"total seconds     : {total:.3f}")
        print(f"throughput        : {callbacks/total if total>0 else 0:.3f} req/s")
        print(f"formula max diff  : {max_formula_diff:.3e}")
        print(f"flat RMSE ratio   : {cand_rmse/max(base_rmse,1e-12):.6f}")
        print(f"mean proxy gain   : {100*mean_proxy_gain:+.2f}%")
        print(f"mean trend gain   : {mean_trend_gain:+.6f}")

        passed = (
            accepted == N_REQUESTS
            and callbacks == N_REQUESTS
            and duplicates == 0
            and errors == 0
            and not submit_errors
            and max_formula_diff <= 1e-8
            and cand_rmse < base_rmse
            and mean_proxy_gain > 0
        )
        print(f"V8-MSF ONLINE GATE: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
