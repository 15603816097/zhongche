from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from src.data_cleaner import clean_sequence
from v9_analog_multiscale_diagnostic import evaluate_rows, proxy_gain

TARGETS = [
    "vibration_rms",
    "temperature_c",
    "current_a",
    "speed_rpm",
    "acoustic_db",
    "pressure_kpa",
]

UNTOUCHED = [
    TARGETS.index("vibration_rms"),
    TARGETS.index("current_a"),
    TARGETS.index("speed_rpm"),
    TARGETS.index("acoustic_db"),
]


def json_safe_float(value):
    """Convert CSV values to strict JSON values; NaN/Inf become null."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def call_api(port: int, payload: dict):
    started = time.perf_counter()
    resp = requests.post(
        f"http://127.0.0.1:{port}/predict",
        json=payload,
        timeout=180,
    )
    elapsed = time.perf_counter() - started
    data = resp.json()

    if resp.status_code != 200:
        raise RuntimeError(f"port={port} HTTP={resp.status_code}")
    if data.get("code") != 0:
        raise RuntimeError(
            f"port={port} code={data.get('code')} message={data.get('message')}"
        )

    predictions = data.get("predictions", [])
    if len(predictions) != 96:
        raise RuntimeError(f"port={port} prediction_count={len(predictions)}")

    arr = np.asarray(
        [[row["values"][c] for c in TARGETS] for row in predictions],
        dtype=np.float64,
    )
    if arr.shape != (96, 6):
        raise RuntimeError(f"port={port} shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"port={port} contains NaN/Inf")
    return arr, elapsed


def main() -> int:
    truth_rows = []
    v8_rows = []
    v9_rows = []
    anchors = []
    times_v8 = []
    times_v9 = []

    print("=" * 112)
    print("V8 vs V9-SAFE - 5 OFFICIAL SEQUENCES")
    print("=" * 112)

    for seq_num in range(1, 6):
        name = f"sequence{seq_num:04d}"
        hdf = pd.read_csv(f"data/raw/{name}/history.csv")
        fdf = pd.read_csv(f"data/raw/{name}/future.csv")

        missing_history_values = int(
            (~np.isfinite(hdf[TARGETS].to_numpy(dtype=np.float64))).sum()
        )

        history = [
            {
                "step": i,
                "values": {c: json_safe_float(row[c]) for c in TARGETS},
            }
            for i, row in hdf.iterrows()
        ]
        payload = {
            "requestId": f"AB_RUNTIME_{name}",
            "history_length": len(history),
            "forecast_horizon": 96,
            "target_columns": TARGETS,
            "history": history,
        }

        p8, t8 = call_api(8800, payload)
        p9, t9 = call_api(8801, payload)

        untouched_diff = float(
            np.max(np.abs(p9[:, UNTOUCHED] - p8[:, UNTOUCHED]))
        )
        if untouched_diff >= 1e-9:
            raise RuntimeError(
                f"{name}: untouched targets changed: {untouched_diff}"
            )

        print(
            f"{name} | V8={t8:.3f}s V9={t9:.3f}s "
            f"extra={t9 - t8:+.3f}s "
            f"untouched_diff={untouched_diff:.12f} "
            f"json_nulls={missing_history_values}"
        )

        # Match API-side preprocessing for the anchor used in trend metrics.
        hdf_clean = clean_sequence(hdf[TARGETS])
        anchor = hdf_clean[TARGETS].iloc[-1].to_numpy(dtype=np.float64)

        truth = fdf[TARGETS].iloc[:96].to_numpy(dtype=np.float64)
        if truth.shape != (96, 6):
            raise RuntimeError(f"{name}: future shape={truth.shape}")
        if not np.isfinite(truth).all():
            raise RuntimeError(f"{name}: future.csv contains NaN/Inf")

        truth_rows.append(truth)
        v8_rows.append(p8)
        v9_rows.append(p9)
        anchors.append(anchor)
        times_v8.append(t8)
        times_v9.append(t9)

    truth = np.stack(truth_rows)
    v8 = np.stack(v8_rows)
    v9 = np.stack(v9_rows)
    anchors_arr = np.stack(anchors)

    rmse8 = float(np.sqrt(np.mean((truth - v8) ** 2)))
    rmse9 = float(np.sqrt(np.mean((truth - v9) ** 2)))

    print("\n" + "=" * 112)
    print("PER TARGET")
    print("=" * 112)

    proxy8 = []
    proxy9 = []
    for j, name in enumerate(TARGETS):
        b = evaluate_rows(
            truth[:, :, j],
            v8[:, :, j],
            anchors_arr[:, j],
        )
        c = evaluate_rows(
            truth[:, :, j],
            v9[:, :, j],
            anchors_arr[:, j],
        )
        ratio = float(c["rmse"] / max(b["rmse"], 1e-12))
        gain = float(proxy_gain(b, c))
        trend_gain = float(c["trend_core"] - b["trend_core"])
        proxy8.append(float(b["proxy_loss"]))
        proxy9.append(float(c["proxy_loss"]))
        print(
            f"{name:16s} rmse_ratio={ratio:.6f} "
            f"proxy_gain={100 * gain:+.2f}% "
            f"trend_gain={trend_gain:+.4f}"
        )

    mean_proxy8 = float(np.mean(proxy8))
    mean_proxy9 = float(np.mean(proxy9))
    mean_proxy_gain = float(
        (mean_proxy8 - mean_proxy9) / max(abs(mean_proxy8), 1e-12)
    )

    print("\n" + "=" * 112)
    print("FINAL RUNTIME REPRODUCTION")
    print("=" * 112)
    print(f"flat RMSE V8        : {rmse8:.6f}")
    print(f"flat RMSE V9-Safe   : {rmse9:.6f}")
    print(f"flat RMSE ratio     : {rmse9 / max(rmse8, 1e-12):.6f}")
    print(f"mean proxy V8       : {mean_proxy8:.6f}")
    print(f"mean proxy V9-Safe  : {mean_proxy9:.6f}")
    print(f"mean proxy gain     : {100 * mean_proxy_gain:+.2f}%")
    print(f"mean V8 time        : {np.mean(times_v8):.3f}s")
    print(f"mean V9-Safe time   : {np.mean(times_v9):.3f}s")
    print(
        f"mean extra time     : "
        f"{np.mean(times_v9) - np.mean(times_v8):+.3f}s"
    )

    if not (rmse9 < rmse8):
        raise SystemExit("FAIL: V9-Safe flat RMSE did not improve")
    if mean_proxy_gain <= 0.01:
        raise SystemExit("FAIL: mean proxy gain <= 1%")

    print("\n5-SEQUENCE RUNTIME TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
