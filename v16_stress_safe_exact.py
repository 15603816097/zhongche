import pickle

import numpy as np

from config import MODEL_DIR
import v16_stress_blend_diagnostic as base


PREP_ARTIFACT = MODEL_DIR / "val_pred_candidate_v15.pkl"
SAFE_INPUT = MODEL_DIR / "v15_safe_exact_for_v16.npz"
SAFE_OUTPUT = MODEL_DIR / "v16_stress_safe_exact.npz"
EXPECTED = np.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 0.5], dtype=np.float64)


def main():
    if not PREP_ARTIFACT.exists():
        raise FileNotFoundError(
            f"缺少 {PREP_ARTIFACT}，请先运行 python train_v15_safe_exact.py"
        )

    with open(PREP_ARTIFACT, "rb") as f:
        prep = pickle.load(f)

    if not bool(prep.get("passed", False)):
        raise RuntimeError("V15 safe exact integration gate 未通过，禁止继续 stress。")

    alphas = np.asarray(prep.get("alphas", []), dtype=np.float64)
    if alphas.shape != EXPECTED.shape or not np.allclose(alphas, EXPECTED, atol=1e-12):
        raise RuntimeError(
            f"safe alpha 不一致: actual={alphas.tolist()} expected={EXPECTED.tolist()}"
        )

    np.savez_compressed(
        SAFE_INPUT,
        alphas=alphas,
        passed=np.asarray(True),
    )

    base.V15_PATH = SAFE_INPUT
    base.OUT_PATH = SAFE_OUTPUT
    base.main()


if __name__ == "__main__":
    main()
