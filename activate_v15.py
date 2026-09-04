import pickle
import shutil
from pathlib import Path

from config import MODEL_DIR, TARGET_COLUMNS
from src.v8_runtime import (
    PCA_MODEL_PATH,
    PCA_PREPROCESS_PATH,
    ROBUST_PCA_MODEL_PATH,
    ROBUST_PCA_PREPROCESS_PATH,
    load_pca_runtime,
    load_robust_pca_runtime,
    v15_alphas,
    v8_parameters,
)


CURRENT_CONFIG = MODEL_DIR / "ensemble_config.pkl"
CANDIDATE_CONFIG = MODEL_DIR / "ensemble_config_candidate_v15.pkl"
BACKUP_CONFIG = MODEL_DIR / "ensemble_config_before_v15.pkl"


def _require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def main():
    print("=" * 92)
    print("Activate V15 official candidate")
    print("=" * 92)

    for path in (
        CURRENT_CONFIG,
        CANDIDATE_CONFIG,
        PCA_MODEL_PATH,
        PCA_PREPROCESS_PATH,
        ROBUST_PCA_MODEL_PATH,
        ROBUST_PCA_PREPROCESS_PATH,
    ):
        _require(path)

    with open(CURRENT_CONFIG, "rb") as f:
        current = pickle.load(f)
    with open(CANDIDATE_CONFIG, "rb") as f:
        candidate = pickle.load(f)

    if int(current.get("version", -1)) != 8:
        raise RuntimeError(
            f"激活 V15 前要求当前线上仍为 V8，实际 version={current.get('version')}"
        )
    if int(candidate.get("version", -1)) != 15:
        raise RuntimeError(
            f"候选版本不是 V15: version={candidate.get('version')}"
        )
    if not bool(candidate.get("candidate_passed_local_gate", False)):
        raise RuntimeError("V15 candidate_passed_local_gate=False，禁止激活")
    if not bool(candidate.get("v15_v16_stress_validated", False)):
        raise RuntimeError("V15 未记录 V16 stress validation，禁止激活")
    if str(candidate.get("trajectory_model", "")) != "pca_xgb_source_aware_hf_robust_blend_v15":
        raise RuntimeError(
            f"未知 V15 trajectory_model: {candidate.get('trajectory_model')}"
        )

    weights, gains, sources, windows = v8_parameters(candidate)
    alphas = v15_alphas(candidate)

    if len(alphas) != len(TARGET_COLUMNS):
        raise RuntimeError("V15 robust alpha 数量错误")
    if not any(float(x) > 1e-12 for x in alphas):
        raise RuntimeError("V15 robust alpha 全为0，禁止激活")

    # 真正反序列化两套 PCA 模型，避免切配置后才发现模型损坏。
    load_pca_runtime()
    load_robust_pca_runtime()

    if not BACKUP_CONFIG.exists():
        shutil.copy2(CURRENT_CONFIG, BACKUP_CONFIG)
        print(f"已备份当前 V8 配置: {BACKUP_CONFIG}")
    else:
        print(f"保留已有 V15 前备份: {BACKUP_CONFIG}")

    active = dict(candidate)
    active["candidate_only"] = False
    active["activated_for_official"] = True

    tmp_path = CURRENT_CONFIG.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(active, f)
    tmp_path.replace(CURRENT_CONFIG)

    print("V15 已激活到 models/ensemble_config.pkl")
    print(f"version       : {active.get('version')}")
    print(f"trajectory    : {active.get('trajectory_model')}")
    print(f"val RMSE      : {active.get('validation_rmse')}")
    print(f"val Proxy     : {active.get('validation_proxy_loss')}")
    print(f"DiffCorr      : {active.get('validation_diff_corr')}")
    print(f"PeakF1        : {active.get('validation_peak_f1')}")
    print(f"VolFit        : {active.get('validation_volatility_fit')}")
    print(f"PCA weights   : {weights.tolist()}")
    print(f"robust alphas : {alphas.tolist()}")
    print(f"HF gains      : {gains.tolist()}")
    print(f"HF sources    : {sources}")
    print(f"HF windows    : {windows.tolist()}")
    print("\n下一步：重启 API，然后运行 python smoke_test_v15.py。")


if __name__ == "__main__":
    main()
