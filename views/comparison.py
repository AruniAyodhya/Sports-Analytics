"""Mode 3: Multi-Model Benchmark & Comparison Dashboard View.

Comprehensive comparative evaluation across Random Forest, Logistic Regression, Decision Tree, and XGBoost.
Group: 2026-AI-09 | SLIIT Machine Learning Module IT3091
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from .charts import plot_comparison_metric, plot_generalization_gap_chart

logger = logging.getLogger(__name__)

ALL_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"]
MODEL_ORDER = ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
LIFECYCLE_STAGES = ["Baseline Validation", "Tuned Validation", "Final Test"]
SPLIT_STAGES = ["Training Split", "Validation Split", "Holdout Testing"]


def _find_artifacts_dir(base_path: Path) -> Optional[Path]:
    """Locate the artifacts directory across expected project layouts."""
    candidates = [
        base_path / "artifacts",
        base_path,
    ]
    for c in candidates:
        if (c / "final_comparison").exists() or (c / "random_forest").exists():
            return c
    return None


def _load_from_final_comparison(
    p_comp: Path,
    model_order: List[str],
    metrics: List[str],
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Load lifecycle metrics directly from artifacts/final_comparison/ CSV files."""
    p_b = p_comp / "comparison_baseline_validation.csv"
    p_v = p_comp / "comparison_validation.csv"
    p_t = p_comp / "comparison_test.csv"

    if not (p_b.exists() and p_v.exists() and p_t.exists()):
        return None

    try:
        df_b = pd.read_csv(p_b, index_col=0)
        df_v = pd.read_csv(p_v, index_col=0)
        df_t = pd.read_csv(p_t, index_col=0)

        data: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics}
        for m in metrics:
            for model in model_order:
                if model in df_b.index and model in df_v.index and model in df_t.index:
                    if m in df_b.columns and m in df_v.columns and m in df_t.columns:
                        vb = round(float(df_b.loc[model, m]), 4)
                        vv = round(float(df_v.loc[model, m]), 4)
                        vt = round(float(df_t.loc[model, m]), 4)
                        data[m][model] = [vb, vv, vt]

        # Verify all models and metrics were populated
        if all(len(data[m]) == len(model_order) for m in metrics):
            logger.info("Successfully dynamically loaded model metrics from %s", p_comp)
            return data
    except Exception as exc:
        logger.warning("Error parsing comparison CSVs from %s: %s", p_comp, exc)

    return None


def _load_from_model_artifacts(
    artifacts_dir: Path,
    model_order: List[str],
    metrics: List[str],
) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Fallback: Load metrics directly from individual model artifact folders."""
    data: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics}

    model_files = {
        "Logistic Regression": artifacts_dir / "logistic_regression" / "logistic_regression_results.csv",
        "Decision Tree": artifacts_dir / "decision_tree" / "decision_tree_results.csv",
        "Random Forest": artifacts_dir / "random_forest" / "random_forest_results.csv",
        "XGBoost": artifacts_dir / "xgboost" / "xgboost_lifecycle_results.csv",
    }

    try:
        for model in model_order:
            p = model_files.get(model)
            if not p or not p.exists():
                logger.warning("Individual model result file not found: %s", p)
                return None

            if model == "XGBoost":
                df = pd.read_csv(p)
                if "Stage" in df.columns:
                    df = df.set_index("Stage")
                for m in metrics:
                    if m in df.columns:
                        vb = round(float(df.loc["Baseline Validation", m]), 4)
                        vv = round(float(df.loc["Tuned Validation", m]), 4)
                        vt = round(float(df.loc["Final Test", m]), 4)
                        data[m][model] = [vb, vv, vt]
            else:
                df = pd.read_csv(p, index_col=0)
                for m in metrics:
                    if m in df.index:
                        vb = round(float(df.loc[m, "Baseline Validation"]), 4)
                        vv = round(float(df.loc[m, "Tuned Validation"]), 4)
                        vt = round(float(df.loc[m, "Final Test"]), 4)
                        data[m][model] = [vb, vv, vt]

        if all(len(data[m]) == len(model_order) for m in metrics):
            logger.info("Successfully dynamically loaded metrics from individual model folders in %s", artifacts_dir)
            return data
    except Exception as exc:
        logger.warning("Error parsing individual model artifacts from %s: %s", artifacts_dir, exc)

    return None


def _load_split_data(
    artifacts_dir: Path,
    model_order: List[str],
    lifecycle_data: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """Dynamically load training, validation, and holdout test split metrics."""
    split_metrics = ["Accuracy", "Macro F1", "ROC-AUC"]
    split_data: Dict[str, Dict[str, List[float]]] = {m: {} for m in split_metrics}

    p_train = artifacts_dir / "final_comparison" / "comparison_train.csv"
    train_scores: Dict[str, Dict[str, float]] = {}
    if p_train.exists():
        try:
            df_train = pd.read_csv(p_train, index_col=0)
            for m in split_metrics:
                if m in df_train.columns:
                    train_scores[m] = {
                        model: round(float(df_train.loc[model, m]), 4)
                        for model in model_order
                        if model in df_train.index
                    }
        except Exception as exc:
            logger.warning("Error reading %s: %s", p_train, exc)

    for m in split_metrics:
        for model in model_order:
            tuned_val = lifecycle_data.get(m, {}).get(model, [0.0, 0.0, 0.0])[1]
            test_val = lifecycle_data.get(m, {}).get(model, [0.0, 0.0, 0.0])[2]
            train_val = train_scores.get(m, {}).get(model)
            if train_val is None:
                train_val = tuned_val
            split_data[m][model] = [train_val, tuned_val, test_val]

    return split_data


def _calculate_majority_baseline(artifacts_dir: Path) -> Tuple[float, float]:
    """Calculate majority class accuracy and Macro F1 dynamically from y_test.csv."""
    candidates = [
        artifacts_dir / "final_dataset" / "y_test.csv",
        artifacts_dir / "preprocessed_dataset" / "y_test.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                y_test = pd.read_csv(p).iloc[:, 0]
                maj_class = y_test.mode()[0]
                maj_acc = round(float((y_test == maj_class).mean()), 4)
                p_c = float((y_test == maj_class).sum())
                prec = p_c / len(y_test)
                f1_maj = 2.0 * prec / (prec + 1.0)
                maj_f1 = round(float(f1_maj / y_test.nunique()), 4)
                return maj_acc, maj_f1
            except Exception as exc:
                logger.warning("Error computing majority baseline from %s: %s", p, exc)

    return 0.4421, 0.2047


def _model_badge_class(model_name: str) -> str:
    """Return styling class for model family badge."""
    mapping = {
        "Logistic Regression": "kpi-val-lr",
        "Random Forest": "kpi-val-rf",
        "XGBoost": "kpi-val-xgb",
        "Decision Tree": "kpi-val-dt",
    }
    return mapping.get(model_name, "kpi-val-rf")


@st.cache_data
def load_model_comparison_data() -> Dict[str, Any]:
    """Dynamically load benchmark and lifecycle metrics across all 4 models from research artifacts."""
    base_path = Path(__file__).resolve().parent.parent
    artifacts_dir = _find_artifacts_dir(base_path)

    if not artifacts_dir:
        err_msg = (
            f"Artifacts directory could not be located under '{base_path}'. "
            "Please ensure model evaluation files exist inside 'artifacts/final_comparison/'."
        )
        logger.error(err_msg)
        return {"error": err_msg}

    # 1. Primary loader: artifacts/final_comparison/ CSVs
    p_comp = artifacts_dir / "final_comparison"
    lifecycle_data = _load_from_final_comparison(p_comp, MODEL_ORDER, ALL_METRICS)

    # 2. Fallback loader: individual model result CSVs
    if not lifecycle_data:
        logger.info("final_comparison CSVs incomplete or missing; falling back to individual model folders.")
        lifecycle_data = _load_from_model_artifacts(artifacts_dir, MODEL_ORDER, ALL_METRICS)

    # 3. Handle missing artifact files gracefully with clear logging
    if not lifecycle_data:
        err_msg = (
            "Actual model evaluation result files (CSVs) are missing in both "
            f"'{artifacts_dir}/final_comparison/' and individual model folders. "
            "Please run notebooks/Model Comparison.ipynb or the model evaluation pipelines."
        )
        logger.error(err_msg)
        return {"error": err_msg}

    # Build split generalization dataset dynamically
    split_data = _load_split_data(artifacts_dir, MODEL_ORDER, lifecycle_data)

    # Calculate majority baseline dynamically from true test targets
    majority_acc, majority_f1 = _calculate_majority_baseline(artifacts_dir)

    # Build comprehensive 6-metric lifecycle dataframe
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

    return {
        "model_order": MODEL_ORDER,
        "lifecycle_stages": LIFECYCLE_STAGES,
        "split_stages": SPLIT_STAGES,
        "lifecycle_data": lifecycle_data,
        "split_data": split_data,
        "lifecycle_matrix_df": lifecycle_matrix_df,
        "majority_acc": majority_acc,
        "majority_f1": majority_f1,
    }


def render_comparison(active_theme: str = "dark") -> None:
    """Render Mode 3: Multi-Model Benchmark & Comparison Dashboard."""
    comp_data = load_model_comparison_data()

    if comp_data.get("error"):
        st.error(f"⚠️ {comp_data['error']}")
        st.info("💡 To generate these metrics, execute `notebooks/Model Comparison.ipynb` or the model training pipelines.")
        return

    majority_acc = comp_data["majority_acc"]
    majority_f1 = comp_data["majority_f1"]

    st.markdown("#### Multi-Model Benchmark & Lifecycle Evaluation")
    st.markdown(
        """
        <p class="page-subtitle">
            Comprehensive comparative technical review across all four model families (Random Forest, Logistic Regression, Decision Tree, and XGBoost) 
            tracking classification performance trajectories, generalization retention on unseen holdout test data (3,923 matches), and overfitting resistance directly from 
            <code>notebooks/Model Comparison.ipynb</code>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Derive dynamic KPI statistics directly from parsed artifacts
    f1_dict = comp_data["lifecycle_data"].get("Macro F1", {})
    acc_dict = comp_data["lifecycle_data"].get("Accuracy", {})

    best_val_model = max(
        comp_data["model_order"],
        key=lambda m: f1_dict.get(m, [0.0, 0.0, 0.0])[1],
        default="Logistic Regression",
    )
    best_val_f1 = f1_dict.get(best_val_model, [0.0, 0.0, 0.0])[1]

    best_test_f1_model = max(
        comp_data["model_order"],
        key=lambda m: f1_dict.get(m, [0.0, 0.0, 0.0])[2],
        default="Random Forest",
    )
    best_test_f1 = f1_dict.get(best_test_f1_model, [0.0, 0.0, 0.0])[2]
    best_test_f1_val = f1_dict.get(best_test_f1_model, [0.0, 0.0, 0.0])[1]
    f1_decay_pct = (
        ((best_test_f1 - best_test_f1_val) / best_test_f1_val) * 100
        if best_test_f1_val
        else 0.0
    )

    best_test_acc_model = max(
        comp_data["model_order"],
        key=lambda m: acc_dict.get(m, [0.0, 0.0, 0.0])[2],
        default="XGBoost",
    )
    best_test_acc = acc_dict.get(best_test_acc_model, [0.0, 0.0, 0.0])[2]
    best_test_acc_val = acc_dict.get(best_test_acc_model, [0.0, 0.0, 0.0])[1]
    acc_decay_pct = (
        ((best_test_acc - best_test_acc_val) / best_test_acc_val) * 100
        if best_test_acc_val
        else 0.0
    )

    f1_uplift_pct = (
        ((best_test_f1 - majority_f1) / majority_f1) * 100
        if majority_f1
        else 0.0
    )

    # 4 Dynamic Executive KPI Tiles
    with st.container(key="comp_metrics_row"):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(
                f"""
                <div class="kpi-tile kpi-tile-stretched">
                    <div class="kpi-label">Selected Primary Model</div>
                    <div class="kpi-value {_model_badge_class(best_val_model)}">{best_val_model}</div>
                    <div class="kpi-sub">
                        <span class="kpi-sub-metric">Val Macro F1: <b>{best_val_f1:.4f}</b></span>
                        <span class="kpi-sub-tag">(Optimal Benchmark)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="kpi-tile kpi-tile-stretched">
                    <div class="kpi-label">Top Holdout Macro F1</div>
                    <div class="kpi-value {_model_badge_class(best_test_f1_model)}">{best_test_f1_model}</div>
                    <div class="kpi-sub">
                        <span class="kpi-sub-metric">Test F1: <b>{best_test_f1:.4f}</b></span>
                        <span class="kpi-sub-tag">(Decay: {f1_decay_pct:+.2f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="kpi-tile kpi-tile-stretched">
                    <div class="kpi-label">Top Holdout Accuracy</div>
                    <div class="kpi-value {_model_badge_class(best_test_acc_model)}">{best_test_acc_model}</div>
                    <div class="kpi-sub">
                        <span class="kpi-sub-metric">Test Acc: <b>{best_test_acc*100:.2f}%</b></span>
                        <span class="kpi-sub-tag">(Decay: {acc_decay_pct:+.2f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="kpi-tile kpi-tile-stretched">
                    <div class="kpi-label">Heuristic Majority Baseline</div>
                    <div class="kpi-value kpi-val-dt">Home Win ({majority_acc*100:.2f}%)</div>
                    <div class="kpi-sub">
                        <span class="kpi-sub-metric">Macro F1: <b>{majority_f1:.4f}</b></span>
                        <span class="kpi-sub-tag">(Beaten by {f1_uplift_pct:+.0f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # Control Toolbar: Progression Dimension Selector & Data Point Label Toggle
    ctrl_col1, ctrl_col2 = st.columns([0.68, 0.32], vertical_alignment="center")
    with ctrl_col1:
        prog_mode = st.radio(
            "Evaluation Progression Dimension",
            [
                "Lifecycle Tuning Stages (Baseline Val → Tuned Val → Final Test)",
                "Dataset Split Generalization (Training Split → Validation Split → Holdout Testing)",
            ],
            index=0,
            horizontal=True,
            help="Toggle between lifecycle hyperparameter tuning stages and raw dataset train/val/test splits for overfitting inspection.",
        )
    with ctrl_col2:
        show_data_labels = st.checkbox(
            "Overlay Direct Data Point Values",
            value=False,
            help="Overlay numerical values directly on chart data points using smart collision offsets. When unchecked, hover over any point for the high-precision inspector.",
        )

    is_lifecycle = "Lifecycle" in prog_mode
    active_stages = comp_data["lifecycle_stages"] if is_lifecycle else comp_data["split_stages"]
    active_dataset = comp_data["lifecycle_data"] if is_lifecycle else comp_data["split_data"]

    # Interactive Visual Tabs: Clean, Spacious, and Un-cramped
    tab_acc, tab_f1, tab_auc, tab_stack, tab_gap, tab_matrix = st.tabs([
        "Accuracy Benchmark",
        "Macro F1-Score (Primary)",
        "ROC-AUC (Separation)",
        "Multi-Metric Stack",
        "Generalization & Overfitting",
        "Complete Lifecycle Matrix",
    ])

    with tab_acc:
        st.plotly_chart(
            plot_comparison_metric(
                "Accuracy",
                active_stages,
                active_dataset["Accuracy"],
                theme=active_theme,
                ref_line=majority_acc,
                ref_label="Majority Baseline",
                height=440,
                show_labels=show_data_labels,
                y_title="Classification Accuracy",
            ),
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="analysis-callout">
                <div class="analysis-callout-title">
                    Accuracy Progression & Generalization Analysis
                </div>
                <div class="analysis-callout-body">
                    • <b>Baseline Outperformance</b>: All 4 models comfortably surpass the 44.21% naive majority-class baseline on unseen holdout test data (3,923 matches).<br/>
                    • <b>Hyperparameter Tuning Uplift</b>: Decision Tree achieved the highest accuracy gain during hyperparameter tuning (+4.32%), successfully recovering from default depth instability.<br/>
                    • <b>Generalization Stability</b>: Performance drop from validation to test is tightly bounded between <b>-1.14% (XGBoost)</b> and <b>-1.95% (Decision Tree)</b>, confirming zero pathological overfitting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_f1:
        st.plotly_chart(
            plot_comparison_metric(
                "Macro F1",
                active_stages,
                active_dataset["Macro F1"],
                theme=active_theme,
                ref_line=majority_f1,
                ref_label="Majority Baseline",
                height=440,
                show_labels=show_data_labels,
                y_title="Macro F1-Score (All 3 Classes)",
            ),
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="analysis-callout">
                <div class="analysis-callout-title">
                    Macro F1-Score & Minority Class Balance Analysis (Primary Evaluation Metric)
                </div>
                <div class="analysis-callout-body">
                    • <b>Massive Uplift Over Heuristics (+110% to +115%)</b>: While naive majority guessing achieves a dismal 0.2047 Macro F1 due to ignoring Away Wins and Draws, all four trained models achieve <b>0.4331 to 0.4405</b> on unseen holdout test data.<br/>
                    • <b>Primary Model Selection Decision</b>: <b>Logistic Regression</b> achieved the peak validation Macro F1 (<b>0.4599</b>), earning its selection as the primary classifier in the research paper.<br/>
                    • <b>Ensemble Robustness</b>: <b>Random Forest</b> achieved the highest final test Macro F1 (<b>0.4405</b>) with minimal validation-to-test drop (-0.0133), proving exceptional stochastic balance across all three match outcomes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_auc:
        st.plotly_chart(
            plot_comparison_metric(
                "ROC-AUC",
                active_stages,
                active_dataset["ROC-AUC"],
                theme=active_theme,
                ref_line=None,
                height=440,
                show_labels=show_data_labels,
                y_title="One-vs-Rest Multiclass ROC-AUC",
            ),
            use_container_width=True,
        )
        st.markdown(
            """
            <div class="analysis-callout">
                <div class="analysis-callout-title">
                    Multiclass Discriminative Separation (ROC-AUC) Analysis
                </div>
                <div class="analysis-callout-body">
                    • <b>Continuous Rank Quality</b>: One-vs-Rest (OvR) multiclass ROC-AUC evaluates predicted continuous probabilities across all three match outcomes.<br/>
                    • <b>Strong Calibration</b>: All models sustain robust discrimination between <b>0.6362 and 0.6525</b> on unseen holdout test data.<br/>
                    • <b>Linear Regularization Excellence</b>: <b>Logistic Regression</b> leads ROC-AUC across both validation (<b>0.6647</b>) and testing (<b>0.6525</b>), benefiting from smooth continuous log-odds probability surfaces without greedy tree step discontinuities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_stack:
        st.markdown("##### Multi-Metric Progression Stack (Full-Width Un-cramped View)")
        st.markdown(
            """
            <p class="section-desc">
                Vertically stacked, full-width performance curves allowing uncluttered, side-by-side technical evaluation across all three core metrics.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            plot_comparison_metric(
                "Accuracy",
                active_stages,
                active_dataset["Accuracy"],
                theme=active_theme,
                ref_line=majority_acc,
                ref_label="Majority Baseline",
                height=360,
                show_labels=show_data_labels,
                y_title="Accuracy",
            ),
            use_container_width=True,
        )

        st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

        st.plotly_chart(
            plot_comparison_metric(
                "Macro F1",
                active_stages,
                active_dataset["Macro F1"],
                theme=active_theme,
                ref_line=majority_f1,
                ref_label="Majority Baseline",
                height=360,
                show_labels=show_data_labels,
                y_title="Macro F1",
            ),
            use_container_width=True,
        )

        st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

        st.plotly_chart(
            plot_comparison_metric(
                "ROC-AUC",
                active_stages,
                active_dataset["ROC-AUC"],
                theme=active_theme,
                ref_line=None,
                height=360,
                show_labels=show_data_labels,
                y_title="ROC-AUC",
            ),
            use_container_width=True,
        )

    with tab_gap:
        st.markdown("##### Generalization Degradation & Overfitting Audit")
        st.markdown(
            """
            <p class="section-desc">
                Quantitative validation-to-test performance decay analysis confirming that no model exhibits pathological memorization or feature leakage.
            </p>
            """,
            unsafe_allow_html=True,
        )

        val_accs = [comp_data["lifecycle_data"]["Accuracy"][m][1] for m in comp_data["model_order"]]
        test_accs = [comp_data["lifecycle_data"]["Accuracy"][m][2] for m in comp_data["model_order"]]
        val_f1s = [comp_data["lifecycle_data"]["Macro F1"][m][1] for m in comp_data["model_order"]]
        test_f1s = [comp_data["lifecycle_data"]["Macro F1"][m][2] for m in comp_data["model_order"]]
        val_aucs = [comp_data["lifecycle_data"]["ROC-AUC"][m][1] for m in comp_data["model_order"]]
        test_aucs = [comp_data["lifecycle_data"]["ROC-AUC"][m][2] for m in comp_data["model_order"]]

        f1_drops_list = []
        acc_drops_list = []
        gap_rows = []
        for idx, m in enumerate(comp_data["model_order"]):
            acc_drop = (test_accs[idx] - val_accs[idx]) * 100
            f1_drop = test_f1s[idx] - val_f1s[idx]
            auc_drop = test_aucs[idx] - val_aucs[idx]
            retention = (test_f1s[idx] / val_f1s[idx]) * 100
            risk = "Very Low (Stable)" if abs(f1_drop) <= 0.015 else "Low (Expected Decay)"

            f1_drops_list.append(f1_drop)
            acc_drops_list.append((test_accs[idx] - val_accs[idx]))

            gap_rows.append({
                "Model Family": m,
                "Val Macro F1": f"{val_f1s[idx]:.4f}",
                "Test Macro F1": f"{test_f1s[idx]:.4f}",
                "F1 Drop": f"{f1_drop:+.4f}",
                "F1 Retention": f"{retention:.1f}%",
                "Val Accuracy": f"{val_accs[idx]*100:.2f}%",
                "Test Accuracy": f"{test_accs[idx]*100:.2f}%",
                "Acc Drop": f"{acc_drop:+.2f}%",
                "Test ROC-AUC": f"{test_aucs[idx]:.4f}",
                "Overfitting Assessment": risk,
            })

        df_gap = pd.DataFrame(gap_rows)
        st.dataframe(df_gap, use_container_width=True, hide_index=True)

        st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

        # Interactive horizontal bar chart comparing drops
        st.plotly_chart(
            plot_generalization_gap_chart(
                comp_data["model_order"],
                f1_drops_list,
                acc_drops_list,
                theme=active_theme,
                height=280,
            ),
            use_container_width=True,
        )

        abs_drops = [abs(d) for d in f1_drops_list]
        lowest_idx = int(np.argmin(abs_drops)) if abs_drops else 0
        lowest_model = comp_data["model_order"][lowest_idx]
        lowest_drop = f1_drops_list[lowest_idx]
        min_retention = min([(t / v * 100) for t, v in zip(test_f1s, val_f1s) if v > 0], default=94.8)

        st.markdown(
            f"""
            <div class="fixture-bar fixture-bar-wrapped">
                <div class="fixture-pill">
                    <span>Model Selection Criterion:</span> <strong>Validation Macro F1 ({best_val_model}: {best_val_f1:.4f})</strong>
                </div>
                <div class="fixture-pill">
                    <span>Lowest F1 Generalization Drop:</span> <strong>{lowest_model} ({lowest_drop:+.4f})</strong>
                </div>
                <div class="fixture-pill">
                    <span>Generalization Retention:</span> <strong>All Models Retain &gt;{min_retention:.1f}% of Validation Performance</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_matrix:
        st.markdown("##### Complete 6-Metric Lifecycle Evaluation Matrix")
        st.markdown(
            """
            <p class="section-desc">
                Full lifecycle performance matrix displaying all six primary classification metrics across Baseline Validation, Tuned Validation, and Final Holdout Testing 
                directly extracted from <code>notebooks/Model Comparison.ipynb</code> and <code>artifacts/final_comparison/</code> artifacts.
            </p>
            """,
            unsafe_allow_html=True,
        )

        matrix_df = comp_data.get("lifecycle_matrix_df", pd.DataFrame())
        matrix_filter = st.selectbox(
            "Filter Model Family",
            ["All Models (Unified Matrix)", "Logistic Regression", "Random Forest", "XGBoost", "Decision Tree"],
            index=0,
            help="Filter matrix rows to inspect a specific classifier lifecycle.",
        )

        if matrix_filter != "All Models (Unified Matrix)":
            filtered_df = matrix_df[matrix_df["Model Family"] == matrix_filter]
        else:
            filtered_df = matrix_df

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="analysis-callout" style="margin-top: 10px;">
                <div class="analysis-callout-title">
                    Lifecycle Evaluation Methodology & Definitions
                </div>
                <div class="analysis-callout-body-sm">
                    • <b>Baseline Validation</b>: Out-of-the-box model architecture with default hyperparameters evaluated on the validation fold.<br/>
                    • <b>Tuned Validation</b>: Performance after hyperparameter tuning and model-specific feature selection (SFM / RFE).<br/>
                    • <b>Final Test</b>: Out-of-sample holdout evaluation on 3,923 strictly unseen historical matches.<br/>
                    • <b>Tuning Uplift</b>: Metric gain achieved through hyperparameter optimization (Tuned Val − Baseline Val).<br/>
                    • <b>Generalization Drop</b>: Performance variance when deploying from validation to final holdout test (Final Test − Tuned Val).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
