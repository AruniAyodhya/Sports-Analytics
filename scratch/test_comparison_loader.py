from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"]
MODEL_ORDER = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
LIFECYCLE_STAGES = ["Baseline Validation", "Tuned Validation", "Final Test"]
SPLIT_STAGES = ["Training Split", "Validation Split", "Holdout Testing"]

def load_test():
    base_path = Path('.').resolve()
    artifacts_dir = base_path / "artifacts"
    p_comp = artifacts_dir / "final_comparison"
    
    # 1. Try final_comparison
    p_b = p_comp / "comparison_baseline_validation.csv"
    p_v = p_comp / "comparison_validation.csv"
    p_t = p_comp / "comparison_test.csv"
    
    lifecycle_data = {m: {} for m in ALL_METRICS}
    
    if p_b.exists() and p_v.exists() and p_t.exists():
        df_b = pd.read_csv(p_b, index_col=0)
        df_v = pd.read_csv(p_v, index_col=0)
        df_t = pd.read_csv(p_t, index_col=0)
        for m in ALL_METRICS:
            for model in MODEL_ORDER:
                if model in df_b.index and model in df_v.index and model in df_t.index:
                    if m in df_b.columns and m in df_v.columns and m in df_t.columns:
                        vb = round(float(df_b.loc[model, m]), 4)
                        vv = round(float(df_v.loc[model, m]), 4)
                        vt = round(float(df_t.loc[model, m]), 4)
                        lifecycle_data[m][model] = [vb, vv, vt]
        logger.info("Loaded from final_comparison successfully")
    else:
        logger.warning("final_comparison missing, trying model folders")
    
    # Verify lifecycle_data
    for m in ALL_METRICS:
        print(f"Metric: {m}")
        for model in MODEL_ORDER:
            print(f"  {model}: {lifecycle_data[m].get(model)}")
            
    # Matrix DF
    matrix_rows = []
    for model in MODEL_ORDER:
        for m in ALL_METRICS:
            vals = lifecycle_data.get(m, {}).get(model, [0.0, 0.0, 0.0])
            b_val, v_val, t_val = vals[0], vals[1], vals[2]
            uplift = v_val - b_val
            drop = t_val - v_val
            matrix_rows.append({
                "Model Family": model,
                "Metric": m,
                "Baseline Val": f"{b_val:.4f}",
                "Tuned Val": f"{v_val:.4f}",
                "Final Test": f"{t_val:.4f}",
                "Tuning Uplift": f"{uplift:+.4f}",
                "Generalization Drop": f"{drop:+.4f}",
            })
    lifecycle_matrix_df = pd.DataFrame(matrix_rows)
    print("Matrix rows count:", len(lifecycle_matrix_df))
    print(lifecycle_matrix_df.head(6))

load_test()
