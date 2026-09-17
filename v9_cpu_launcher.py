from __future__ import annotations

import os

# The current server GPU/driver combination may not support the CUDA kernels
# shipped by xgboost-cu12.  V9 itself is CPU-only; only the exact V8 baseline
# needs XGBoost.  Force the already-trained XGBoost models to CPU for this
# offline diagnostic without modifying the production V8 service.
os.environ.setdefault("XGB_DEVICE", "cpu")

import src.inference as inference
import src.v8_runtime as v8_runtime


def _force_xgb_cpu(model, label: str) -> None:
    """Force an XGBoost sklearn/native model (or nested estimators) to CPU."""
    seen = set()

    def visit(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        try:
            obj.set_params(device="cpu")
        except Exception:
            pass

        try:
            booster = obj.get_booster()
            booster.set_param({"device": "cpu", "nthread": max(1, min(8, os.cpu_count() or 1))})
        except Exception:
            pass

        estimators = getattr(obj, "estimators_", None)
        if estimators is not None:
            try:
                for est in estimators:
                    visit(est)
            except TypeError:
                pass

    visit(model)
    print(f"[V9 CPU FALLBACK] {label}: forced device=cpu", flush=True)


def prepare_v8_cpu() -> None:
    # Load the same exact V8 pickle files used by production, then only change
    # runtime device placement.  Model weights/configuration stay untouched.
    _, model_xgb, _, _, cfg = inference.load_models()
    _force_xgb_cpu(model_xgb, "main XGBoost")

    if v8_runtime.v8_enabled(cfg):
        pca_model, _ = v8_runtime.load_pca_runtime()
        _force_xgb_cpu(pca_model, "V8 PCA XGBoost")

    print(
        "[V9 CPU FALLBACK] exact V8 weights retained; only XGBoost inference device changed",
        flush=True,
    )


def main() -> int:
    prepare_v8_cpu()
    import v9_analog_multiscale_diagnostic as diagnostic
    return int(diagnostic.main())


if __name__ == "__main__":
    raise SystemExit(main())
