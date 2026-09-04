import pickle
import shutil
from pathlib import Path

from config import MODEL_DIR


CURRENT_CONFIG = MODEL_DIR / "ensemble_config.pkl"
BACKUP_CONFIG = MODEL_DIR / "ensemble_config_before_v15.pkl"


def main():
    print("=" * 88)
    print("Rollback V15 -> previous V8 config")
    print("=" * 88)

    if not BACKUP_CONFIG.exists():
        raise FileNotFoundError(f"缺少 V15 前备份: {BACKUP_CONFIG}")

    with open(BACKUP_CONFIG, "rb") as f:
        backup = pickle.load(f)
    if int(backup.get("version", -1)) != 8:
        raise RuntimeError(
            f"V15 前备份不是 V8: version={backup.get('version')}"
        )

    tmp_path = CURRENT_CONFIG.with_suffix(".pkl.tmp")
    shutil.copy2(BACKUP_CONFIG, tmp_path)
    tmp_path.replace(CURRENT_CONFIG)

    print("已恢复 V8 ensemble_config.pkl")
    print(f"version    : {backup.get('version')}")
    print(f"trajectory : {backup.get('trajectory_model')}")
    print("请重启 API，使进程重新加载 V8 配置。")


if __name__ == "__main__":
    main()
