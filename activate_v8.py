import pickle
import shutil
from pathlib import Path

from config import MODEL_DIR, TARGET_COLUMNS
from src.v8_runtime import (
    PCA_MODEL_PATH,
    PCA_PREPROCESS_PATH,
    load_pca_runtime,
    v8_parameters,
)


CURRENT_CONFIG = MODEL_DIR / "ensemble_config.pkl"
CANDIDATE_CONFIG = MODEL_DIR / "ensemble_config_candidate_v8.pkl"
BACKUP_CONFIG = MODEL_DIR / "ensemble_config_before_v8.pkl"


def _require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def main():
    print("=" * 88)
    print("Activate V8 official candidate")
    print("=" * 88)

    for path in (
        CURRENT_CONFIG,
        CANDIDATE_CONFIG,
        PCA_MODEL_PATH,
        PCA_PREPROCESS_PATH,
    ):
        _require(path)

    with open(CANDIDATE_CONFIG, "rb") as f:
        candidate = pickle.load(f)

    if int(candidate.get("version", -1)) != 8:
        raise RuntimeError(
            f"候选版本不是 V8: version={candidate.get('version')}"
        )
    if not bool(candidate.get("candidate_passed_local_gate", False)):
        raise RuntimeError("V8 candidate_passed_local_gate=False，禁止激活")
    if str(candidate.get("trajectory_model", "")) != "pca_xgb_source_aware_hf_v1":
        raise RuntimeError(
            f"未知 V8 trajectory_model: {candidate.get('trajectory_model')}"
        )

    weights, gains, sources, windows = v8_parameters(candidate)
    if not any(float(x) > 1e-12 for x in weights):
        raise RuntimeError("V8 PCA 权重全为 0，禁止激活")
    if len(sources) != len(TARGET_COLUMNS):
        raise RuntimeError("V8 高频来源数量错误")

    # 先真正反序列化 PCA 模型/PCA/scaler，避免配置切换后才发现模型文件损坏。
    load_pca_runtime()

    if not BACKUP_CONFIG.exists():
        shutil.copy2(CURRENT_CONFIG, BACKUP_CONFIG)
        print(f"已备份当前线上配置: {BACKUP_CONFIG}")
    else:
        print(f"保留已有 V8 前备份: {BACKUP_CONFIG}")

    active = dict(candidate)
    active["candidate_only"] = False
    active["activated_for_official"] = True

    tmp_path = CURRENT_CONFIG.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(active, f)
    tmp_path.replace(CURRENT_CONFIG)

    print("V8 已激活到 models/ensemble_config.pkl")
    print(f"version     : {active.get('version')}")
    print(f"local gate  : {active.get('candidate_passed_local_gate')}")
    print(f"val RMSE    : {active.get('validation_rmse')}")
    print(f"DiffCorr    : {active.get('validation_diff_corr')}")
    print(f"PeakF1      : {active.get('validation_peak_f1')}")
    print(f"VolFit      : {active.get('validation_volatility_fit')}")
    print(f"PCA weights : {weights.tolist()}")
    print(f"HF gains    : {gains.tolist()}")
    print(f"HF sources  : {sources}")
    print(f"HF windows  : {windows.tolist()}")
    print("\n下一步必须重启 API 进程，旧进程缓存的仍是旧 ensemble_config。")


if __name__ == "__main__":
    main()
