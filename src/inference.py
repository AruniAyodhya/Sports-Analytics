"""Modular Inference Engine for European Soccer Match Outcome Prediction.

Group: 2026-AI-09, SLIIT ML Module IT3091
Anti-Leakage Protocol: Strictly pre-match features only (rolling form, odds, team differentials).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


class SoccerInferenceEngine:
    """Production-grade Inference Engine for European Soccer Match Outcome Prediction.

    Supports dynamic environment base paths (local workstation and Google Colab),
    in-memory caching of artifacts and baseline vectors, feature synthesis for what-if
    simulations, historical test fixture evaluation, and extensible multi-model switching
    across Random Forest, Logistic Regression, Decision Tree, and XGBoost.
    """

    SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
        "random_forest": {
            "name": "Random Forest Classifier (Tuned)",
            "dir": "random_forest",
            "pipeline_file": "random_forest_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
            "threshold_file": "calibrated_draw_threshold.joblib",
            "family": "tree",
        },
        "logistic_regression": {
            "name": "Logistic Regression Classifier (L1 Tuned)",
            "dir": "logistic_regression",
            "pipeline_file": "logistic_regression_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_drop_columns.joblib",
            "threshold_file": "calibrated_draw_threshold.joblib",
            "family": "logistic",
        },
        "decision_tree": {
            "name": "Decision Tree Classifier (Tuned RFE)",
            "dir": "decision_tree",
            "pipeline_file": "decision_tree_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
            "threshold_file": "calibrated_draw_threshold.joblib",
            "family": "tree",
        },
        "xgboost": {
            "name": "XGBoost Classifier (Tuned)",
            "dir": "xgboost",
            "pipeline_file": "xgboost_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
            "threshold_file": "calibrated_draw_threshold.joblib",
            "family": "tree",
        },
    }

    CLEAN_LEAGUE_MAP: Dict[str, str] = {
        "England Premier League": "league_name_England Premier League",
        "France Ligue 1": "league_name_France Ligue 1",
        "Germany 1. Bundesliga": "league_name_Germany 1. Bundesliga",
        "Italy Serie A": "league_name_Italy Serie A",
        "Netherlands Eredivisie": "league_name_Netherlands Eredivisie",
        "Poland Ekstraklasa": "league_name_Poland Ekstraklasa",
        "Portugal Liga ZON Sagres": "league_name_Portugal Liga ZON Sagres",
        "Scotland Premier League": "league_name_Scotland Premier League",
        "Spain LIGA BBVA": "league_name_Spain LIGA BBVA",
        "Switzerland Super League": "league_name_Switzerland Super League",
    }

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        model_type: str = "random_forest",
    ) -> None:
        """Initialize the inference engine with configurable base path and model type."""
        self.base_dir = self._resolve_base_dir(base_dir)
        self.artifacts_dir = (self.base_dir / "artifacts") if (self.base_dir / "artifacts").exists() else self.base_dir
        self.model_type = model_type if model_type in self.SUPPORTED_MODELS else "random_forest"

        # In-memory artifact caches
        self._target_encoder = None
        self._scaler = None
        self._dropped_columns: Dict[str, List[str]] = {}
        self._thresholds: Dict[str, float] = {}
        self._pipeline_cache: Dict[str, Any] = {}
        self._median_baselines: Dict[str, pd.Series] = {}
        self._league_baselines: Dict[str, pd.Series] = {}
        self._test_X_cache: Dict[str, pd.DataFrame] = {}
        self._test_y: Optional[pd.DataFrame] = None
        self._test_metadata: Optional[pd.DataFrame] = None

        # Pre-load core artifacts
        self.load_target_encoder()
        self.load_scaler()
        self.load_dropped_columns(self.model_type)
        self.load_threshold(self.model_type)
        self.load_model(self.model_type)

    @classmethod
    def _resolve_base_dir(cls, base_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve base directory dynamically across Local OS, Colab, or environment variables."""
        if base_dir is not None:
            resolved = Path(base_dir).resolve()
            if resolved.exists():
                return resolved

        env_dir = os.environ.get("SOCCER_BASE_DIR")
        if env_dir and Path(env_dir).exists():
            return Path(env_dir).resolve()

        # Check default parent of src/
        parent_candidate = Path(__file__).resolve().parent.parent
        if (
            (parent_candidate / "artifacts").exists()
            or (parent_candidate / "final_dataset").exists()
            or (parent_candidate / "random_forest").exists()
        ):
            return parent_candidate

        # Check current working directory
        cwd_candidate = Path.cwd().resolve()
        if (
            (cwd_candidate / "artifacts").exists()
            or (cwd_candidate / "final_dataset").exists()
            or (cwd_candidate / "random_forest").exists()
        ):
            return cwd_candidate

        # Check common Google Colab paths
        colab_candidate = Path("/content/Sports Analytics")
        if colab_candidate.exists():
            return colab_candidate

        colab_candidate_cwd = Path("/content")
        if (colab_candidate_cwd / "artifacts").exists() or (colab_candidate_cwd / "final_dataset").exists():
            return colab_candidate_cwd

        return parent_candidate

    # -------------------------------------------------------------------------
    # Artifact Loaders & Caching
    # -------------------------------------------------------------------------

    def load_target_encoder(self):
        """Load and cache the fitted target LabelEncoder.

        Target Encoding Mapping:
            - Class 0: Away Win (away_team_goal > home_team_goal)
            - Class 1: Draw (home_team_goal == away_team_goal)
            - Class 2: Home Win (home_team_goal > away_team_goal)

        Returns:
            LabelEncoder: Fitted scikit-learn LabelEncoder instance for multi-class target inverse transforms.
        """
        if self._target_encoder is None:
            target_enc_path = self.artifacts_dir / "final_dataset" / "target_encoder.joblib"
            if not target_enc_path.exists():
                target_enc_path = self.base_dir / "final_dataset" / "target_encoder.joblib"
            if not target_enc_path.exists():
                raise FileNotFoundError(
                    f"Target encoder artifact missing at {target_enc_path}"
                )
            self._target_encoder = joblib.load(target_enc_path)
        return self._target_encoder

    def load_scaler(self):
        """Load and cache the fitted StandardScaler used for numerical normalization.

        Mathematical & Model Context (L1-Regularized Logistic Regression):
            Unlike tree ensembles that are invariant to monotonic scaling, Logistic Regression
            is deeply sensitive to feature magnitude. In L1-regularized (Lasso) multinomial
            logistic regression:
                min_w  -sum_i sum_k y_ik * log(P(Y=k|x_i)) + lambda * ||w||_1
            features with wide numeric dispersion (e.g. Elo ~1500) would dominate the penalty term
            over tightly bounded metrics (e.g. win rates in [0, 1]).
            
            The StandardScaler normalizes each feature to zero mean and unit variance:
                z = (x - mu) / sigma
            ensuring all candidate predictors compete equitably during sparse coefficient selection.

        Returns:
            StandardScaler: Pre-fitted scikit-learn StandardScaler deserialized from disk.
        """
        if self._scaler is None:
            scaler_path = self.artifacts_dir / "final_dataset" / "logistic_scaler.joblib"
            if not scaler_path.exists():
                scaler_path = self.artifacts_dir / "final_dataset" / "scaler.joblib"
            if not scaler_path.exists():
                scaler_path = self.base_dir / "final_dataset" / "logistic_scaler.joblib"
            if not scaler_path.exists():
                scaler_path = self.base_dir / "final_dataset" / "scaler.joblib"
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
        return self._scaler

    def load_dropped_columns(self, model_type: Optional[str] = None) -> List[str]:
        """Load and cache the Stage-1 correlation dropped columns for a given model.

        Preprocessing Context (Stage-1 Multicollinearity Pruning):
            During offline preprocessing, pairwise Pearson correlation coefficients (|r|) were
            computed across all numeric candidate features. If any two features exhibited |r| >= 0.90,
            the redundant colinear feature was pruned to eliminate matrix collinearity, stabilize
            regression coefficients, and prevent variance inflation in linear and distance models.

        Args:
            model_type: Model identifier key (defaults to active self.model_type).

        Returns:
            List[str]: Column names eliminated during offline feature selection.
        """
        m_type = model_type or self.model_type
        if m_type in self._dropped_columns:
            return self._dropped_columns[m_type]

        model_info = self.SUPPORTED_MODELS.get(m_type, self.SUPPORTED_MODELS["random_forest"])
        model_dir = self.artifacts_dir / model_info["dir"]
        if not model_dir.exists():
            model_dir = self.base_dir / model_info["dir"]

        dropped_path = model_dir / model_info.get("dropped_cols_file", "correlation_dropped_columns.joblib")
        if not dropped_path.exists():
            dropped_path = model_dir / "correlation_dropped_columns.joblib"
        if not dropped_path.exists():
            dropped_path = model_dir / "correlation_drop_columns.joblib"
        if not dropped_path.exists():
            dropped_path = self.artifacts_dir / "random_forest" / "correlation_dropped_columns.joblib"
        if not dropped_path.exists():
            dropped_path = self.base_dir / "random_forest" / "correlation_dropped_columns.joblib"

        if not dropped_path.exists():
            self._dropped_columns[m_type] = []
            return self._dropped_columns[m_type]

        try:
            self._dropped_columns[m_type] = list(joblib.load(dropped_path))
        except Exception as drop_err:
            warnings.warn(
                f"Failed to load dropped columns from {dropped_path}: {drop_err}. Using default empty list.",
                RuntimeWarning,
            )
            self._dropped_columns[m_type] = []
        return self._dropped_columns[m_type]

    def load_threshold(self, model_type: Optional[str] = None) -> float:
        """Load and cache the calibrated optimal draw threshold for a model (defaults to 0.360)."""
        m_type = model_type or self.model_type
        if m_type in self._thresholds:
            return self._thresholds[m_type]

        model_info = self.SUPPORTED_MODELS.get(m_type, self.SUPPORTED_MODELS["random_forest"])
        model_dir = self.artifacts_dir / model_info["dir"]
        if not model_dir.exists():
            model_dir = self.base_dir / model_info["dir"]
        threshold_path = model_dir / model_info.get("threshold_file", "calibrated_draw_threshold.joblib")

        if threshold_path.exists():
            try:
                thr = float(joblib.load(threshold_path))
            except Exception:
                thr = 0.360
        else:
            thr = 0.360

        self._thresholds[m_type] = thr
        return self._thresholds[m_type]

    def load_model(self, model_type: str = "random_forest"):
        """Dynamically load and cache the requested model pipeline artifact.

        Includes a robust self-healing fallback mechanism: if an artifact file is missing,
        corrupted, or fails to deserialize (e.g. cross-version XGBoost input stream corruption),
        it automatically attempts to re-create the pipeline on the fly or gracefully falls back
        without halting the application.
        """
        if model_type in self._pipeline_cache:
            self.model_type = model_type
            return self._pipeline_cache[model_type]

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Choose from {list(self.SUPPORTED_MODELS.keys())}"
            )

        model_info = self.SUPPORTED_MODELS[model_type]
        model_dir = self.artifacts_dir / model_info["dir"]
        if not model_dir.exists():
            model_dir = self.base_dir / model_info["dir"]
        pipeline_path = model_dir / model_info["pipeline_file"]

        # Search for any candidate .joblib file containing 'pipeline' if primary path is missing
        if not pipeline_path.exists() and model_dir.exists():
            joblib_candidates = list(model_dir.glob("*pipeline*.joblib"))
            if joblib_candidates:
                pipeline_path = joblib_candidates[0]

        pipeline = None
        if pipeline_path.exists():
            try:
                pipeline = joblib.load(pipeline_path)
            except Exception as load_err:
                warnings.warn(
                    f"Artifact notice: Failed to load '{model_info['name']}' from {pipeline_path}: {load_err}. "
                    "Initiating self-healing pipeline recovery.",
                    RuntimeWarning,
                )
                pipeline = None

        if pipeline is None:
            # Self-healing attempt: rebuild missing or corrupted model artifact
            pipeline = self._recover_or_rebuild_pipeline(model_type, pipeline_path)

        if pipeline is None:
            # Graceful fallback: fall back to Random Forest to prevent app crashing
            if model_type != "random_forest":
                warnings.warn(
                    f"Unable to load or rebuild '{model_type}' pipeline artifact. "
                    "Gracefully falling back to 'random_forest'.",
                    RuntimeWarning,
                )
                fallback_pipeline = self.load_model("random_forest")
                self._pipeline_cache[model_type] = fallback_pipeline
                self.model_type = "random_forest"
                return fallback_pipeline
            else:
                raise FileNotFoundError(
                    f"Model pipeline artifact for '{model_info['name']}' not found at: {pipeline_path}\n"
                    f"Please ensure models are trained and placed in their respective directories."
                )

        self._pipeline_cache[model_type] = pipeline
        self.model_type = model_type
        return pipeline

    def _recover_or_rebuild_pipeline(
        self, model_type: str, pipeline_path: Path
    ) -> Optional[Any]:
        """Self-healing fallback mechanism to rebuild missing or corrupted model pipelines."""
        if model_type == "xgboost":
            return self._rebuild_xgboost_pipeline(pipeline_path)
        return None

    def _rebuild_xgboost_pipeline(self, pipeline_path: Path) -> Optional[Any]:
        """Re-train and persist the tuned XGBoost classification pipeline.

        Uses the project's exact hyperparameter architecture and balanced sample weights
        to re-create a healthy, natively deserializable pipeline artifact.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.feature_selection import SelectFromModel
            from sklearn.pipeline import Pipeline
            from sklearn.utils.class_weight import compute_sample_weight

            final_dir = self.artifacts_dir / "final_dataset"
            if not final_dir.exists():
                final_dir = self.base_dir / "final_dataset"

            tree_train_path = final_dir / "X_train_tree.csv"
            y_train_path = final_dir / "y_train.csv"

            if not tree_train_path.exists() or not y_train_path.exists():
                warnings.warn(
                    f"Training datasets for XGBoost recovery missing at {tree_train_path} or {y_train_path}.",
                    RuntimeWarning,
                )
                return None

            X_train_tree = pd.read_csv(tree_train_path)
            market_cols = [c for c in X_train_tree.columns if c.startswith("market_")]
            X_clean = X_train_tree.drop(columns=market_cols)

            dropped_cols = self.load_dropped_columns("xgboost")
            X_train_corr = X_clean.drop(columns=dropped_cols, errors="ignore")

            y_train = pd.read_csv(y_train_path).iloc[:, 0].astype(int)
            sample_weights = compute_sample_weight("balanced", y=y_train)

            selector = SelectFromModel(
                estimator=XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    importance_type="gain",
                    random_state=42,
                    n_jobs=-1,
                ),
                threshold=-np.inf,
                max_features=50,
            )

            classifier = XGBClassifier(
                subsample=0.7,
                reg_lambda=2.0,
                reg_alpha=0.1,
                n_estimators=200,
                min_child_weight=7,
                max_depth=2,
                learning_rate=0.08,
                gamma=0.05,
                colsample_bytree=0.7,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )

            pipe = Pipeline([
                ("selector", selector),
                ("classifier", classifier),
            ])

            pipe.fit(
                X_train_corr,
                y_train,
                selector__sample_weight=sample_weights,
                classifier__sample_weight=sample_weights,
            )

            # Persist the rebuilt pipeline and companion artifacts to disk
            pipeline_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                joblib.dump(pipe, pipeline_path)
                selector_path = pipeline_path.parent / "xgb_feature_selector.joblib"
                joblib.dump(pipe.named_steps["selector"], selector_path)
                selected_cols = X_train_corr.columns[pipe.named_steps["selector"].get_support()].tolist()
                feats_path = pipeline_path.parent / "xgboost_selected_features.joblib"
                joblib.dump(selected_cols, feats_path)
            except Exception as save_err:
                warnings.warn(f"Could not persist rebuilt XGBoost model to disk: {save_err}")

            return pipe
        except Exception as rebuild_err:
            warnings.warn(f"Failed to rebuild XGBoost pipeline: {rebuild_err}", RuntimeWarning)
            return None

    # -------------------------------------------------------------------------
    # Baseline Vectors & Test Fixtures
    # -------------------------------------------------------------------------

    def get_median_baseline(self, model_type: Optional[str] = None) -> pd.Series:
        """Compute and cache in-memory the median feature baseline vector.

        Uses unscaled features for tree-based models and standardized features for Logistic Regression.
        """
        m_type = model_type or self.model_type
        family = self.SUPPORTED_MODELS.get(m_type, {}).get("family", "tree")

        if family in self._median_baselines:
            return self._median_baselines[family]

        final_dir = self.artifacts_dir / "final_dataset"
        if not final_dir.exists():
            final_dir = self.base_dir / "final_dataset"

        if family == "logistic":
            train_path = final_dir / "X_train_logistic.csv"
            if not train_path.exists():
                train_path = final_dir / "X_train_model.csv"
        else:
            train_path = final_dir / "X_train_tree.csv"
            if not train_path.exists():
                train_path = final_dir / "X_train_model.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset missing at {train_path}")

        df_train = pd.read_csv(train_path)
        self._median_baselines[family] = df_train.median()
        return self._median_baselines[family]

    def get_league_baseline(self, league_name: Optional[str] = None) -> pd.Series:
        """Retrieve the empirical median baseline vector for a specific league competition.

        Returns unscaled features from X_train_tree.csv so domain feature synthesis remains
        natural, grounded, and bounded before model-family standardization.
        """
        if not self._league_baselines:
            final_dir = self.artifacts_dir / "final_dataset"
            if not final_dir.exists():
                final_dir = self.base_dir / "final_dataset"

            tree_train_path = final_dir / "X_train_tree.csv"
            if not tree_train_path.exists():
                tree_train_path = final_dir / "X_train_model.csv"

            if tree_train_path.exists():
                df_tree = pd.read_csv(tree_train_path)
                global_med = df_tree.median()
                self._league_baselines["__global__"] = global_med

                for clean_name, col_name in self.CLEAN_LEAGUE_MAP.items():
                    if col_name in df_tree.columns:
                        subset = df_tree[df_tree[col_name] == 1.0]
                        if len(subset) > 0:
                            self._league_baselines[clean_name] = subset.median()
                        else:
                            self._league_baselines[clean_name] = global_med
                    else:
                        self._league_baselines[clean_name] = global_med
            else:
                fallback = self.get_median_baseline("random_forest")
                self._league_baselines["__global__"] = fallback
                for clean_name in self.CLEAN_LEAGUE_MAP:
                    self._league_baselines[clean_name] = fallback

        if league_name and league_name in self._league_baselines:
            return self._league_baselines[league_name]
        return self._league_baselines.get("__global__", self.get_median_baseline("random_forest"))

    def get_test_data(self, model_type: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and cache the historical holdout test set with zero temporal leakage.

        Chronological Splitting Protocol:
            - Training Set: Seasons 2008/2009 to 2013/2014 (18,179 fixtures) - Used for model fitting.
            - Validation Set: Season 2014/2015 (3,932 fixtures) - Used for tuning & threshold calibration.
            - Holdout Test Set: Season 2015/2016 (3,923 fixtures) - Strictly isolated final evaluation set.

        Data Architecture:
            Loads either standardized features (`X_test_logistic.csv`) for linear classifiers
            or raw unscaled features (`X_test_tree.csv`) for tree-based ensemble estimators.

        Args:
            model_type: Identifier of the model to determine feature matrix family.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (X_test feature matrix, y_test target matrix).
        """
        m_type = model_type or self.model_type
        family = self.SUPPORTED_MODELS.get(m_type, {}).get("family", "tree")

        if family not in self._test_X_cache or self._test_y is None:
            final_dir = self.artifacts_dir / "final_dataset"
            if not final_dir.exists():
                final_dir = self.base_dir / "final_dataset"

            if family == "logistic":
                test_x_path = final_dir / "X_test_logistic.csv"
                if not test_x_path.exists():
                    test_x_path = final_dir / "X_test_model.csv"
            else:
                test_x_path = final_dir / "X_test_tree.csv"
                if not test_x_path.exists():
                    test_x_path = final_dir / "X_test_model.csv"

            test_y_path = final_dir / "y_test.csv"

            if not test_x_path.exists() or not test_y_path.exists():
                raise FileNotFoundError(
                    f"Test dataset files missing at {test_x_path} or {test_y_path}"
                )

            self._test_X_cache[family] = pd.read_csv(test_x_path)
            self._test_y = pd.read_csv(test_y_path)

        return self._test_X_cache[family], self._test_y

    def get_test_metadata(self) -> Optional[pd.DataFrame]:
        """Load test fixture metadata (league, stage, odds) for human inspection."""
        if self._test_metadata is None:
            meta_path = self.artifacts_dir / "final_dataset" / "test_metadata.csv"
            if not meta_path.exists():
                meta_path = self.base_dir / "final_dataset" / "test_metadata.csv"
            if meta_path.exists():
                self._test_metadata = pd.read_csv(meta_path)
            else:
                raw_split_path = self.artifacts_dir / "splitted_dataset" / "test.csv"
                if not raw_split_path.exists():
                    raw_split_path = self.base_dir / "splitted_dataset" / "test.csv"
                if raw_split_path.exists():
                    self._test_metadata = pd.read_csv(raw_split_path)
        return self._test_metadata

    # -------------------------------------------------------------------------
    # Scaling Helpers & Calibration
    # -------------------------------------------------------------------------

    def scale_single_feature(self, col: str, val: float) -> float:
        """Standardize an individual numeric feature value using the fitted StandardScaler.

        Mathematical Formulation:
            z = (x - mu) / sigma

        Notes:
            Essential for Logistic Regression inference, mapping unscaled physical metrics
            (such as Elo differences or points-per-match) into unit-variance coordinates.
            Guarantees alignment with the training feature distribution.

        Args:
            col: Name of the feature column in the scaler's feature_names_in_ space.
            val: Raw unscaled float value.

        Returns:
            float: Standardized z-score (or original val if column was not standardized).
        """
        scaler = self.load_scaler()
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            if col in scaler.feature_names_in_:
                idx = list(scaler.feature_names_in_).index(col)
                scale = scaler.scale_[idx]
                if scale > 0:
                    return float((val - scaler.mean_[idx]) / scale)
        return float(val)

    def unscale_single_feature(self, col: str, val: float) -> float:
        """Invert feature standardization back to human-interpretable domain units.

        Mathematical Formulation:
            x = z * sigma + mu

        Notes:
            Employed in UI metadata extraction and tactical telemetry displays where
            standardized features in Logistic Regression test vectors must be presented
            in human-readable form (e.g., season stage, points, or goal differences).

        Args:
            col: Name of the feature column.
            val: Standardized z-score.

        Returns:
            float: Unscaled value in original physical domain units.
        """
        scaler = self.load_scaler()
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            if col in scaler.feature_names_in_:
                idx = list(scaler.feature_names_in_).index(col)
                return float(val * scaler.scale_[idx] + scaler.mean_[idx])
        return float(val)

    @staticmethod
    def predict_with_draw_threshold(p: np.ndarray, threshold: float = 0.360) -> int:
        """Classify 3-class outcome probabilities using the calibrated draw threshold boundary.

        Problem Context & Threshold Calibration Mechanics:
            In association football match forecasting, draws (Class 1) occur in ~25% of matches
            but suffer from high tactical stochasticity. Standard multi-class argmax decision rules
            (y_hat = argmax_k P(Y=k)) frequently suffer from severe draw under-prediction because
            home win (Class 2) and away win (Class 0) probabilities routinely surpass 0.35.

            The calibrated decision boundary implements a prioritized draw threshold:
                y_hat = 1 (Draw)        if P(Draw) >= theta_draw
                y_hat = 2 (Home Win)    else if P(Home Win) >= P(Away Win)
                y_hat = 0 (Away Win)    otherwise

            The default threshold (theta_draw = 0.360) is tuned via 5-fold cross-validation
            on the chronological validation set (2014/15) to maximize multi-class Macro F1.

        Args:
            p: 1D or 2D array of predicted class probabilities ordered as [Away Win, Draw, Home Win].
            threshold: Calibrated draw probability threshold (default: 0.360).

        Returns:
            int or np.ndarray: Predicted class integer index (0, 1, or 2).
        """
        p = np.asarray(p)
        if p.ndim == 1:
            if p[1] >= threshold:
                return 1
            return 2 if p[2] >= p[0] else 0
        return np.where(p[:, 1] >= threshold, 1, np.where(p[:, 2] >= p[:, 0], 2, 0))

    # -------------------------------------------------------------------------
    # Feature Synthesis & Preprocessing
    # -------------------------------------------------------------------------

    def synthesize_simulation_features(
        self, input_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """Synthesize a complete fixture feature vector overlaying user inputs on empirical baselines.

        Data Preprocessing & Synthesis Architecture:
            The simulation engine performs deterministic multi-stage feature reconstruction,
            transforming user-facing tactical telemetry (e.g. league, stage, win-rate differentials,
            points-per-match, recent points momentum) into a complete, model-compliant feature row.
            
            Multi-Phase Transformation Workflow:
                Phase 1 (Baseline Prior): Ingests the empirical median feature vector corresponding
                    to the selected league competition, pre-populating historical metrics and one-hot
                    league indicators (`league_*`).
                Phase 2 (Temporal Progression & Fixture Congestion): Computes estimated rest intervals
                    via dynamic schedule decay:
                        rest_days = 14.0 - (stage / 38.0) * 8.0
                    Applies stage maturity scaling factor to model early-season variance vs late-season
                    polarization:
                        stage_factor = 0.75 + 0.50 * (stage / 38.0)
                Phase 3 (Anti-Leakage Safeguard): Excludes in-match post-kickoff stats and decouples
                    betting market odds, ensuring purely athletic pre-match feature representations.
                Phase 4 (Differential Propagation): Projects team strength onto multi-timescale
                    rolling horizons:
                        - Elo rating differentials: derived from win rate differentials and goal margins.
                        - Scoring and conceding rates: venue-specific home/away scoring distributions.
                        - Multi-window rolling form: 3-match surge, 5-match form, 10-match stability.
                Phase 5 (Statistical Normalization): For linear models (Logistic Regression),
                    standardizes numeric features using the offline fitted StandardScaler:
                        z = (x - mu) / sigma
                    For tree models, preserves native unscaled physical units.
                Phase 6 (Multicollinearity Filtering): Automatically drops redundant colinear
                    features (Pearson |r| >= 0.90) identified during offline feature selection.
                Phase 7 (Schema & Order Alignment): Reindexes feature columns to strictly match
                    the trained scikit-learn pipeline's expected input dimensions (`feature_names_in_`).

        Args:
            input_dict: Dictionary containing simulation parameters (league, stage, form differences).

        Returns:
            pd.DataFrame: Single-row feature DataFrame formatted for model prediction.
        """
        family = self.SUPPORTED_MODELS.get(self.model_type, {}).get("family", "tree")
        is_logistic = bool(family == "logistic")

        # 1. League-Specific Empirical Unscaled Baseline
        selected_league = input_dict.get("league", "England Premier League")
        base_series = self.get_league_baseline(selected_league).copy()
        row = base_series.to_dict()

        # Set one-hot league indicator features
        for clean_name, col_name in self.CLEAN_LEAGUE_MAP.items():
            if col_name in row:
                row[col_name] = 1.0 if clean_name == selected_league else 0.0

        # 2. Stage & Fixture Congestion / Rest Days
        stage_val = float(input_dict.get("stage", 19))
        row["stage"] = stage_val
        rest_days = 14.0 - (stage_val / 38.0) * 8.0  # ~14 days at start down to ~6 days late-season
        row["home_days_since_last_match"] = rest_days
        row["away_days_since_last_match"] = rest_days
        row["rest_days_difference"] = 0.0

        # Stage maturity factor: early season (stage 1) differentials are less established;
        # late season (stage 38) differentials are fully polarized.
        stage_factor = 0.75 + 0.50 * (stage_val / 38.0)

        # 3. Market Consensus Odds (Strictly Decoupled)
        # Anti-Leakage / Pure Sports Analytics Requirement:
        # Betting market odds (such as Bet365 Home, Draw, Away odds) do NOT drive or alter
        # the match outcome or prediction probabilities. Predictions rely strictly and exclusively
        # on pure team performance features (form, stats, Elo, and differentials).

        # 4. User Sliders & Form Differentials (Dynamically Extracted from Centralized Registry)
        from .features import extract_simulator_inputs

        core_inputs = extract_simulator_inputs(input_dict)
        ppm_diff = core_inputs["points_per_match_difference"]
        recent_pts_diff = core_inputs["recent_points_difference"]
        goals_diff = core_inputs["goals_for_difference"]
        win_rate_diff = core_inputs["win_rate_difference"]

        # League baseline metrics (scoring pace, draw rate, baseline Elo and goal difference)
        base_hg = float(base_series.get("home_goals_for_per_match", 1.30))
        base_ag = float(base_series.get("away_goals_for_per_match", 1.30))
        base_hdr = float(base_series.get("home_draw_rate", 0.26))
        base_adr = float(base_series.get("away_draw_rate", 0.26))
        base_hwr = float(base_series.get("home_win_rate", 0.38))
        base_awr = float(base_series.get("away_win_rate", 0.38))

        league_pace = base_hg / 1.30
        league_draw_factor = 1.0 - (base_hdr - 0.25) * 1.5
        league_elo_offset = float(base_series.get("elo_difference", 0.0))
        league_gd_offset = float(base_series.get("goals_for_difference", 0.0))
        league_wr_offset = float(base_series.get("win_rate_difference", 0.0))

        # Short-term form momentum
        recent_momentum = (recent_pts_diff - 0.8) * 0.06

        # Elo rating differentials (purely from historical team strength & form)
        elo_diff = (
            league_elo_offset
            + ((win_rate_diff + recent_momentum) * 250.0)
            + (goals_diff * 90.0)
            + (ppm_diff * 70.0)
            + (base_hwr - 0.38) * 120.0
        ) * stage_factor
        elo_diff = float(np.clip(elo_diff, -450.0, 450.0))
        row["elo_difference"] = elo_diff
        row["home_elo_before"] = float(np.clip(base_series.get("home_elo_before", 1500.0) + elo_diff / 2.0, 1100.0, 1950.0))
        row["away_elo_before"] = float(np.clip(base_series.get("away_elo_before", 1500.0) - elo_diff / 2.0, 1100.0, 1950.0))

        # Goal differentials & individual scoring rates
        eff_goal_diff = float(
            (league_gd_offset + goals_diff * league_pace + (ppm_diff - 0.5) * 0.35 + recent_momentum) * stage_factor
        )
        eff_goal_diff = float(np.clip(eff_goal_diff, -3.0, 3.0))

        row["goals_for_difference"] = eff_goal_diff
        row["goals_against_difference"] = -eff_goal_diff
        row["venue_goals_for_difference"] = eff_goal_diff * 0.85
        row["venue_goals_against_difference"] = -eff_goal_diff * 0.85

        # Individual scoring and conceding rates
        row["home_goals_for_per_match"] = float(np.clip(base_hg + eff_goal_diff * 0.45, 0.15, 3.5))
        row["away_goals_for_per_match"] = float(np.clip(base_ag - eff_goal_diff * 0.45, 0.15, 3.5))
        row["home_goals_against_per_match"] = float(np.clip(base_hg - eff_goal_diff * 0.45, 0.15, 3.5))
        row["away_goals_against_per_match"] = float(np.clip(base_ag + eff_goal_diff * 0.45, 0.15, 3.5))

        row["away_away_goals_for_per_match"] = float(np.clip(base_ag - eff_goal_diff * 0.40, 0.15, 3.5))
        row["away_away_goals_against_per_match"] = float(np.clip(base_ag + eff_goal_diff * 0.40, 0.15, 3.5))
        row["home_home_goals_against_per_match"] = float(np.clip(base_hg - eff_goal_diff * 0.40, 0.15, 3.5))

        # Rolling 10-match goals
        row["goals_for_last10_difference"] = eff_goal_diff * 0.95
        row["goals_against_last10_difference"] = -eff_goal_diff * 0.95
        row["home_goals_for_last10"] = float(np.clip(base_hg + eff_goal_diff * 0.45, 0.1, 4.0))
        row["home_goals_against_last10"] = float(np.clip(base_hg - eff_goal_diff * 0.45, 0.1, 4.0))
        row["away_goals_for_last10"] = float(np.clip(base_ag - eff_goal_diff * 0.45, 0.1, 4.0))
        row["away_goals_against_last10"] = float(np.clip(base_ag + eff_goal_diff * 0.45, 0.1, 4.0))

        # Win rate differentials & class probability distributions
        eff_wr_diff = float(
            (league_wr_offset + (win_rate_diff + (base_hwr - 0.38) * 0.5) * league_draw_factor + (ppm_diff - 0.5) * 0.30 + recent_momentum) * stage_factor
        )
        eff_wr_diff = float(np.clip(eff_wr_diff, -0.9, 0.9))

        row["win_rate_difference"] = eff_wr_diff
        row["venue_win_rate_difference"] = eff_wr_diff * 0.85
        row["draw_rate_difference"] = 0.0

        h_wr = float(np.clip(base_hwr + eff_wr_diff * 0.45, 0.05, 0.90))
        a_wr = float(np.clip(base_awr - eff_wr_diff * 0.45, 0.05, 0.90))
        h_dr = base_hdr
        a_dr = base_adr

        row["home_win_rate"] = h_wr
        row["away_win_rate"] = a_wr
        row["home_draw_rate"] = h_dr
        row["away_draw_rate"] = a_dr
        row["home_loss_rate"] = float(np.clip(1.0 - h_wr - h_dr, 0.05, 0.90))
        row["away_loss_rate"] = float(np.clip(1.0 - a_wr - a_dr, 0.05, 0.90))
        row["home_home_win_rate"] = float(np.clip(h_wr + 0.06, 0.05, 0.95))
        row["home_home_draw_rate"] = h_dr
        row["away_away_win_rate"] = float(np.clip(a_wr - 0.06, 0.05, 0.95))
        row["away_away_draw_rate"] = a_dr

        # Points-Per-Match & Rolling Points Form
        pts10_diff = float(np.clip(ppm_diff * 0.9 + (eff_wr_diff * 0.5), -2.8, 2.8))
        row["points_last10_difference"] = pts10_diff
        row["home_points_last10"] = float(np.clip(1.36 + pts10_diff * 0.45, 0.1, 2.9))
        row["away_points_last10"] = float(np.clip(1.36 - pts10_diff * 0.45, 0.1, 2.9))

        # Recent 5-Match & 3-Match Points Form
        pts5_diff = float(np.clip(recent_pts_diff * 0.85 + ppm_diff * 0.15, -3.0, 3.0))
        row["points_last5_difference"] = pts5_diff
        row["points_last3_difference"] = pts5_diff * 0.9
        row["home_points_last5"] = float(np.clip(1.36 + pts5_diff * 0.45, 0.1, 2.9))
        row["away_points_last5"] = float(np.clip(1.36 - pts5_diff * 0.45, 0.1, 2.9))
        row["home_points_last3"] = float(np.clip(1.36 + pts5_diff * 0.40, 0.1, 2.9))
        row["away_points_last3"] = float(np.clip(1.36 - pts5_diff * 0.40, 0.1, 2.9))

        # Short-term Goal Conceding Differentials (Key driver for Logistic Regression)
        row["goals_against_last5_difference"] = float(np.clip(-pts5_diff * 0.4 - eff_goal_diff * 0.4, -3.0, 3.0))
        row["goals_for_last5_difference"] = float(np.clip(pts5_diff * 0.4 + eff_goal_diff * 0.4, -3.0, 3.0))
        row["goals_against_last3_difference"] = float(np.clip(-pts5_diff * 0.35, -3.0, 3.0))
        row["goals_for_last3_difference"] = float(np.clip(pts5_diff * 0.35, -3.0, 3.0))

        row["home_goals_for_last5"] = float(np.clip(base_hg + pts5_diff * 0.25, 0.1, 3.5))
        row["home_goals_against_last5"] = float(np.clip(base_hg - pts5_diff * 0.25, 0.1, 3.5))
        row["away_goals_for_last5"] = float(np.clip(base_ag - pts5_diff * 0.25, 0.1, 3.5))
        row["away_goals_against_last5"] = float(np.clip(base_ag + pts5_diff * 0.25, 0.1, 3.5))

        row["home_goals_for_last3"] = float(np.clip(base_hg + pts5_diff * 0.20, 0.1, 3.5))
        row["home_goals_against_last3"] = float(np.clip(base_hg - pts5_diff * 0.20, 0.1, 3.5))
        row["away_goals_for_last3"] = float(np.clip(base_ag - pts5_diff * 0.20, 0.1, 3.5))
        row["away_goals_against_last3"] = float(np.clip(base_ag + pts5_diff * 0.20, 0.1, 3.5))

        df_result = pd.DataFrame([row])

        # 5. Standardize Numerical Features for Logistic Regression
        # Machine Learning Context:
        # Logistic Regression computes linear decision boundaries: z_k = w_k^T x + b_k, followed
        # by the softmax activation P(Y=k|x) = exp(z_k) / sum_j exp(z_j).
        # Because the model is trained with an L1 (Lasso) sparsity penalty, unstandardized features
        # with high numerical scales would distort the gradient and shrink disproportionately.
        # Here we project the synthesized feature vector into standard normal space (z-scores)
        # matching the offline fitted StandardScaler (mean=0, variance=1).
        if is_logistic:
            scaler = self.load_scaler()
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                for col in scaler.feature_names_in_:
                    if col in df_result.columns:
                        df_result[col] = self.scale_single_feature(col, float(df_result[col].iloc[0]))

        # 6. Drop Stage-1 Multicollinear Features
        dropped_cols = self.load_dropped_columns(self.model_type)
        df_reduced = df_result.drop(columns=dropped_cols, errors="ignore")

        # 7. Exact Schema Alignment with Pipeline Feature Names
        pipeline = self.load_model(self.model_type)
        if hasattr(pipeline, "feature_names_in_"):
            expected_features = [str(f) for f in pipeline.feature_names_in_]
            df_reduced.columns = [str(c) for c in df_reduced.columns]
            df_reduced = df_reduced.reindex(columns=expected_features)

        return df_reduced

    # -------------------------------------------------------------------------
    # Inference Methods
    # -------------------------------------------------------------------------

    def predict_simulation(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform match outcome prediction on a user-synthesized what-if simulation.

        Returns:
            Dict containing predicted_class_id, predicted_label, confidence,
            calibrated probability distribution, and features count.
        """
        pipeline = self.load_model(self.model_type)
        encoder = self.load_target_encoder()
        threshold = self.load_threshold(self.model_type)

        # Synthesize feature vector
        X_feature = self.synthesize_simulation_features(input_dict)

        # Predict probabilities with warning suppression for internal sklearn transformer output types
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            probabilities = pipeline.predict_proba(X_feature)[0]

        # Apply calibrated draw threshold
        pred_id = int(self.predict_with_draw_threshold(probabilities, threshold))
        pred_label = str(encoder.inverse_transform([pred_id])[0])

        class_labels = list(encoder.classes_)  # ['Away Win', 'Draw', 'Home Win']
        prob_dict = {
            label: float(probabilities[idx])
            for idx, label in enumerate(class_labels)
        }
        confidence = round(float(probabilities[pred_id]) * 100, 2)

        return {
            "predicted_class_id": pred_id,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "features_used_count": X_feature.shape[1],
            "draw_threshold": threshold,
        }

    def predict_test_match(self, match_index: int) -> Dict[str, Any]:
        """Evaluate a historical match fixture from the holdout test set.

        Returns prediction, probability distribution, ground-truth outcome,
        classification correctness status, and unscaled fixture metadata.
        """
        test_X, test_y = self.get_test_data(self.model_type)
        total_matches = len(test_X)

        if not (0 <= match_index < total_matches):
            raise IndexError(
                f"Test match index {match_index} out of bounds (0 to {total_matches - 1})."
            )

        pipeline = self.load_model(self.model_type)
        encoder = self.load_target_encoder()
        dropped_cols = self.load_dropped_columns(self.model_type)
        threshold = self.load_threshold(self.model_type)

        raw_row = test_X.iloc[[match_index]]

        # Extract fixture metadata
        metadata = self._extract_fixture_metadata(raw_row, match_index)

        # Drop market features if pipeline doesn't use them
        market_cols = [c for c in raw_row.columns if c.startswith("market_")]
        X_fixture = raw_row.drop(columns=market_cols, errors="ignore")
        X_fixture = X_fixture.drop(columns=dropped_cols, errors="ignore")

        if hasattr(pipeline, "feature_names_in_"):
            expected_feats = [str(f) for f in pipeline.feature_names_in_]
            X_fixture.columns = [str(c) for c in X_fixture.columns]
            X_fixture = X_fixture.reindex(columns=expected_feats)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            probabilities = pipeline.predict_proba(X_fixture)[0]
        pred_id = int(self.predict_with_draw_threshold(probabilities, threshold))
        pred_label = str(encoder.inverse_transform([pred_id])[0])

        actual_id = int(test_y.iloc[match_index, 0])
        actual_label = str(encoder.inverse_transform([actual_id])[0])
        is_correct = bool(pred_id == actual_id)

        class_labels = list(encoder.classes_)
        prob_dict = {
            label: float(probabilities[idx])
            for idx, label in enumerate(class_labels)
        }
        confidence = round(float(probabilities[pred_id]) * 100, 2)

        return {
            "match_index": match_index,
            "predicted_class_id": pred_id,
            "predicted_label": pred_label,
            "actual_class_id": actual_id,
            "actual_label": actual_label,
            "is_correct": is_correct,
            "confidence": confidence,
            "probabilities": prob_dict,
            "metadata": metadata,
            "draw_threshold": threshold,
        }

    def _extract_fixture_metadata(self, row: pd.DataFrame, match_index: int) -> Dict[str, Any]:
        """Extract human-readable, unscaled metadata from a holdout test match fixture row.

        Notes on Logistic Regression Feature Inversion:
            Because Logistic Regression operates on standardized inputs (`X_test_logistic.csv`),
            raw fixture features exist in z-score coordinates. For human inspection in the UI
            dashboard (e.g. stage number, match odds), standardized attributes are dynamically
            inverted back to real-world domain values via `unscale_single_feature` (x = z * sigma + mu).

        Args:
            row: Single-fixture DataFrame row from the holdout test set.
            match_index: Positional integer index of the test fixture.

        Returns:
            Dict[str, Any]: Extracted metadata containing league name, stage, and betting market payouts.
        """
        # Try retrieving rich metadata from test_metadata.csv
        test_meta = self.get_test_metadata()
        if test_meta is not None and match_index < len(test_meta):
            m_row = test_meta.iloc[match_index]
            l_name = m_row.get("league_name", "European League")
            stg = int(m_row.get("stage", 1))
            b_h = m_row.get("B365H")
            b_d = m_row.get("B365D")
            b_a = m_row.get("B365A")
            return {
                "league": str(l_name),
                "stage": stg,
                "odds": {
                    "Home Win (B365H)": round(float(b_h), 2) if pd.notna(b_h) else None,
                    "Draw (B365D)": round(float(b_d), 2) if pd.notna(b_d) else None,
                    "Away Win (B365A)": round(float(b_a), 2) if pd.notna(b_a) else None,
                },
            }

        # Fallback to row-level extraction
        league_detected = "European League"
        for clean_name, col in self.CLEAN_LEAGUE_MAP.items():
            if col in row.columns and float(row[col].values[0]) == 1.0:
                league_detected = clean_name
                break

        family = self.SUPPORTED_MODELS.get(self.model_type, {}).get("family", "tree")
        raw_stage = row["stage"].values[0] if "stage" in row.columns else 1
        unscaled_stage = int(round(self.unscale_single_feature("stage", float(raw_stage)))) if family == "logistic" else int(round(float(raw_stage)))

        b365h = float(row["B365H"].values[0]) if "B365H" in row.columns and pd.notna(row["B365H"].values[0]) else None
        b365d = float(row["B365D"].values[0]) if "B365D" in row.columns and pd.notna(row["B365D"].values[0]) else None
        b365a = float(row["B365A"].values[0]) if "B365A" in row.columns and pd.notna(row["B365A"].values[0]) else None

        return {
            "league": league_detected,
            "stage": max(1, min(38, unscaled_stage)),
            "odds": {
                "Home Win (B365H)": round(b365h, 2) if b365h else None,
                "Draw (B365D)": round(b365d, 2) if b365d else None,
                "Away Win (B365A)": round(b365a, 2) if b365a else None,
            },
        }
