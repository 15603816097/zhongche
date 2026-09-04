from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests

from config import MODEL_DIR, TARGET_COLUMNS


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "external_data" / "corpus" / "official_finetune_v1.npz"
OUTPUT_PATH = MODEL_DIR / "deep" / "v81_http_stress.json"

API_HOST = os.getenv("V81_STRESS_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("V81_STRESS_API_PORT", "18882"))
CALLBACK_HOST = os.getenv("V81_STRESS_CALLBACK_HOST", "127.0.0.1")
CALLBACK_PORT = int(os.getenv("V81_STRESS_CALLBACK_PORT", "18883"))
NUM_REQUESTS = int(os.getenv("V81_STRESS_REQUESTS", "50"))
SUBMIT_WORKERS = int(os.getenv("V81_STRESS_SUBMIT_WORKERS", "10"))
WAIT_SECONDS = float(os.getenv("V81_STRESS_WAIT_SECONDS", "240"))
API_KEY = os.getenv("API_KEY", "").strip()

API_BASE = f"http://{API_HOST}:{API_PORT}"
CALLBACK_URL = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"

_CALLBACKS: list[dict] = []
_CALLBACK_LOCK = threading.Lock()
_CALLBACK_EVENT = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if self.path != "/callback":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {"_invalid_json": raw.decode("utf-8", errors="replace")}

        with _CALLBACK_LOCK:
            _CALLBACKS.append(body)
            if len(_CALLBACKS) >= NUM_REQUESTS:
                _CALLBACK_EVENT.set()

        payload = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        return


def _history_payload(history: np.ndarray, request_id: str) -> dict:
    return {
        "requestId": request_id,
        "forecast_horizon": 96,
        "history_length": 512,
        "target_columns": list(TARGET_COLUMNS),
        "history": [
            {
                "step": int(i),
                "values": {
                    name: float(history[i, j])
                    for j, name in enumerate(TARGET_COLUMNS)
                },
            }
            for i in range(history.shape[0])
        ],
        "callback_url": CALLBACK_URL,
        "callback_token": f"token-{request_id}",
    }


def _headers() -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if API_KEY:
        out["Authorization"] = f"Bearer {API_KEY}"
    return out


def _submit_one(history: np.ndarray, k: int) -> dict:
    request_id = f"V81_STRESS_{k:03d}"
    payload = _history_payload(history, request_id)
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            headers=_headers(),
            timeout=15,
        )
        elapsed = time.perf_counter() - started
        try:
            body = resp.json()
        except Exception:
            body = {"_raw": resp.text[:500]}
        accepted = bool(
            resp.status_code == 200
            and body.get("code") == 0
            and body.get("message") == "accepted"
            and body.get("predictions") == []
        )
        return {
            "request_id": request_id,
            "status_code": resp.status_code,
            "accepted": accepted,
            "elapsed": elapsed,
            "body": body,
        }
    except Exception as exc:
        return {
            "request_id": request_id,
            "status_code": None,
            "accepted": False,
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _validate_callback(body: dict) -> tuple[bool, str, str | None]:
    if not isinstance(body, dict):
        return False, "body is not object", None
    callback_token = body.get("callback_token")
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False, "results schema", None
    one = results[0]
    if not isinstance(one, dict):
        return False, "result entry schema", None
    request_id = one.get("request_id")
    if not isinstance(request_id, str) or not request_id.startswith("V81_STRESS_"):
        return False, "request_id schema", None
    if callback_token != f"token-{request_id}":
        return False, "callback_token mismatch", request_id
    data = one.get("data")
    if not isinstance(data, dict):
        return False, "data schema", request_id
    if data.get("code") != 0 or data.get("message") != "success":
        return False, f"callback code/message: {data.get('code')} {data.get('message')}", request_id
    pred = data.get("predictions")
    if not isinstance(pred, list) or len(pred) != 96:
        return False, "predictions length", request_id
    for step, row in enumerate(pred):
        if not isinstance(row, dict) or int(row.get("step", -1)) != step:
            return False, f"bad step {step}", request_id
        values = row.get("values")
        if not isinstance(values, dict) or set(values) != set(TARGET_COLUMNS):
            return False, f"values schema at step {step}", request_id
        vals = np.asarray([values[name] for name in TARGET_COLUMNS], dtype=np.float64)
        if not np.isfinite(vals).all():
            return False, f"nonfinite prediction at step {step}", request_id
    return True, "OK", request_id


def _get_json(path: str) -> tuple[int, dict]:
    resp = requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=10)
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text[:500]}
    return resp.status_code, body


def main() -> int:
    if not CORPUS_PATH.is_file():
        print(f"missing corpus: {CORPUS_PATH}")
        return 2

    data = np.load(CORPUS_PATH, allow_pickle=False)
    X = data["X"].astype(np.float64, copy=False)
    center = data["center"].astype(np.float64, copy=False)
    scale = data["scale"].astype(np.float64, copy=False)
    histories = X * scale[:, None, :] + center[:, None, :]

    print("=" * 108)
    print("V8.1 REAL HTTP ASYNC + CALLBACK STRESS")
    print("=" * 108)
    print(f"api                 : {API_BASE}")
    print(f"callback receiver   : {CALLBACK_URL}")
    print(f"requests            : {NUM_REQUESTS}")
    print(f"submit concurrency  : {SUBMIT_WORKERS}")
    print("prediction workers  : service-side current app setting (expected 2)")
    print("callback rule       : exact verified nested results/request_id/data schema")

    health_before_status, health_before = _get_json("/health")
    candidate_status, candidate = _get_json("/candidate")
    health_before_ok = bool(health_before_status == 200 and health_before.get("status") == "ok")
    candidate_ok = bool(
        candidate_status == 200
        and candidate.get("v81_temperature_enabled") is True
        and abs(float(candidate.get("v81_temperature_weight", -1.0)) - 0.15) < 1e-12
    )
    print(f"health before       : {'PASS' if health_before_ok else 'FAIL'}")
    print(f"candidate endpoint  : {'PASS' if candidate_ok else 'FAIL'}")

    callback_server = ThreadingHTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    callback_thread = threading.Thread(target=callback_server.serve_forever, daemon=True)
    callback_thread.start()

    started_all = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=SUBMIT_WORKERS) as ex:
            submissions = list(
                ex.map(
                    lambda k: _submit_one(histories[k % len(histories)], k),
                    range(NUM_REQUESTS),
                )
            )

        submit_elapsed = time.perf_counter() - started_all
        accepted = [x for x in submissions if x.get("accepted")]
        accept_times = np.asarray([x["elapsed"] for x in submissions], dtype=np.float64)
        accept_p95 = float(np.percentile(accept_times, 95)) if len(accept_times) else float("nan")
        accept_max = float(np.max(accept_times)) if len(accept_times) else float("nan")
        print(
            f"HTTP accepted       : {len(accepted)}/{NUM_REQUESTS} "
            f"submit_elapsed={submit_elapsed:.3f}s p95={accept_p95:.3f}s max={accept_max:.3f}s"
        )

        if len(accepted) == NUM_REQUESTS:
            _CALLBACK_EVENT.wait(timeout=WAIT_SECONDS)

        callback_elapsed = time.perf_counter() - started_all
        with _CALLBACK_LOCK:
            callbacks = list(_CALLBACKS)
    finally:
        callback_server.shutdown()
        callback_server.server_close()
        callback_thread.join(timeout=5)

    valid_count = 0
    callback_errors: list[str] = []
    callback_ids: list[str] = []
    for body in callbacks:
        ok, reason, request_id = _validate_callback(body)
        if request_id is not None:
            callback_ids.append(request_id)
        if ok:
            valid_count += 1
        else:
            callback_errors.append(f"{request_id or '?'}: {reason}")

    expected_ids = {f"V81_STRESS_{k:03d}" for k in range(NUM_REQUESTS)}
    seen_ids = set(callback_ids)
    missing_ids = sorted(expected_ids - seen_ids)
    duplicate_count = len(callback_ids) - len(seen_ids)
    unexpected_ids = sorted(seen_ids - expected_ids)

    health_after_status, health_after = _get_json("/health")
    health_after_ok = bool(health_after_status == 200 and health_after.get("status") == "ok")

    throughput = NUM_REQUESTS / max(callback_elapsed, 1e-9)
    callback_complete = len(callbacks) == NUM_REQUESTS
    schema_ok = valid_count == NUM_REQUESTS and not callback_errors
    ids_ok = not missing_ids and duplicate_count == 0 and not unexpected_ids
    accept_ok = len(accepted) == NUM_REQUESTS

    # Sanity gates, not official scoring thresholds.
    gate = bool(
        health_before_ok
        and candidate_ok
        and accept_ok
        and accept_p95 <= 1.0
        and callback_complete
        and schema_ok
        and ids_ok
        and health_after_ok
        and callback_elapsed <= WAIT_SECONDS
    )

    print("\n" + "-" * 108)
    print(f"accepted requests                  : {len(accepted)}/{NUM_REQUESTS}")
    print(f"accept p95                         : {accept_p95:.3f}s")
    print(f"callbacks received                 : {len(callbacks)}/{NUM_REQUESTS}")
    print(f"callbacks schema-valid             : {valid_count}/{NUM_REQUESTS}")
    print(f"missing callback ids               : {len(missing_ids)}")
    print(f"duplicate callback ids             : {duplicate_count}")
    print(f"unexpected callback ids            : {len(unexpected_ids)}")
    print(f"service health after stress        : {'PASS' if health_after_ok else 'FAIL'}")
    print(f"total until all callbacks          : {callback_elapsed:.3f}s")
    print(f"effective completed throughput     : {throughput:.3f} req/s")
    print(f"V8.1 REAL HTTP STRESS GATE         : {'PASS' if gate else 'REJECT'}")

    if callback_errors:
        print("callback errors (first 5):")
        for x in callback_errors[:5]:
            print("  ", x)
    if missing_ids:
        print("missing ids (first 10):", missing_ids[:10])

    result = {
        "candidate": "app_v81:app",
        "requests": NUM_REQUESTS,
        "submit_workers": SUBMIT_WORKERS,
        "health_before_ok": health_before_ok,
        "candidate_endpoint_ok": candidate_ok,
        "accepted": len(accepted),
        "accept_p95_seconds": accept_p95,
        "accept_max_seconds": accept_max,
        "callbacks_received": len(callbacks),
        "callbacks_schema_valid": valid_count,
        "missing_callback_ids": missing_ids,
        "duplicate_callback_ids": duplicate_count,
        "unexpected_callback_ids": unexpected_ids,
        "callback_errors": callback_errors,
        "health_after_ok": health_after_ok,
        "total_seconds_until_callbacks": callback_elapsed,
        "effective_completed_requests_per_second": throughput,
        "gate_pass": gate,
        "note": "Local real-HTTP async/callback stress. app.py and verified callback schema are unchanged.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics                             : {OUTPUT_PATH}")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
