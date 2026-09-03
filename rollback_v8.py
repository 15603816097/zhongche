import shutil

from config import MODEL_DIR


CURRENT_CONFIG = MODEL_DIR / "ensemble_config.pkl"
BACKUP_CONFIG = MODEL_DIR / "ensemble_config_before_v8.pkl"


def main():
    if not BACKUP_CONFIG.exists():
        raise FileNotFoundError(f"缺少 V8 前备份: {BACKUP_CONFIG}")

    shutil.copy2(BACKUP_CONFIG, CURRENT_CONFIG)
    print(f"已回滚: {BACKUP_CONFIG} -> {CURRENT_CONFIG}")
    print("请重启 API 进程，使回滚配置生效。")


if __name__ == "__main__":
    main()
