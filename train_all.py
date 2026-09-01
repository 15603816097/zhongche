from find_best_weight import main as tune_ensemble
from src.trainer import train as train_lgb
from src.trainer_xgb import train as train_xgb


if __name__ == "__main__":
    print("\n[1/3] Train LightGBM")
    train_lgb()

    print("\n[2/3] Train XGBoost")
    train_xgb()

    print("\n[3/3] Tune ensemble")
    tune_ensemble()

    print("\n全部完成。生成/更新的核心文件：")
    print("  models/model_lgb.pkl")
    print("  models/scaler.pkl")
    print("  models/model_xgb.pkl")
    print("  models/scaler_xgb.pkl")
    print("  models/ensemble_config.pkl")
