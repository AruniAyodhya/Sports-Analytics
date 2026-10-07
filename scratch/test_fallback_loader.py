from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"]
MODEL_ORDER = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]

def load_from_individual_models():
    base_path = Path('.').resolve()
    artifacts_dir = base_path / "artifacts"
    
    model_files = {
        "Logistic Regression": artifacts_dir / "logistic_regression" / "logistic_regression_results.csv",
        "Decision Tree": artifacts_dir / "decision_tree" / "decision_tree_results.csv",
        "Random Forest": artifacts_dir / "random_forest" / "random_forest_results.csv",
        "XGBoost": artifacts_dir / "xgboost" / "xgboost_lifecycle_results.csv",
    }
    
    data = {m: {} for m in ALL_METRICS}
    for model in MODEL_ORDER:
        p = model_files.get(model)
        assert p and p.exists(), f"Missing {p}"
        if model == "XGBoost":
            df = pd.read_csv(p)
            if "Stage" in df.columns:
                df = df.set_index("Stage")
            for m in ALL_METRICS:
                vb = round(float(df.loc["Baseline Validation", m]), 4)
                vv = round(float(df.loc["Tuned Validation", m]), 4)
                vt = round(float(df.loc["Final Test", m]), 4)
                data[m][model] = [vb, vv, vt]
        else:
            df = pd.read_csv(p, index_col=0)
            for m in ALL_METRICS:
                vb = round(float(df.loc[m, "Baseline Validation"]), 4)
                vv = round(float(df.loc[m, "Tuned Validation"]), 4)
                vt = round(float(df.loc[m, "Final Test"]), 4)
                data[m][model] = [vb, vv, vt]
                
    print("Fallback successful! RF Acc:", data["Accuracy"]["Random Forest"])
    print("Fallback successful! LR F1:", data["Macro F1"]["Logistic Regression"])
    print("Fallback successful! XGB AUC:", data["ROC-AUC"]["XGBoost"])

load_from_individual_models()
