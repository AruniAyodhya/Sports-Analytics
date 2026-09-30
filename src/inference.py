"""Modular Inference Engine for European Soccer Match Outcome Prediction.

Group: 2026-AI-09, SLIIT ML Module IT3091
Anti-Leakage Protocol: Strictly pre-match features only (rolling form, odds, team differentials).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


class SoccerInferenceEngine:
    """Production-grade Inference Engine for European Soccer Match Outcome Prediction.

    Supports dynamic environment base paths (local workstation and Google Colab),
    in-memory caching of artifacts and baseline vectors, feature synthesis for what-if
    simulations, historical test fixture evaluation, and extensible multi-model switching.
    """

    SUPPORTED_MODELS: Dict[str, Dict[str, str]] = {
        "random_forest": {
            "name": "Random Forest Classifier (Tuned)",
            "dir": "random_forest",
            "pipeline_file": "random_forest_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
        },
        "logistic_regression": {
            "name": "Logistic Regression Classifier",
            "dir": "logistic_regression",
            "pipeline_file": "logistic_regression_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
        },
        "decision_tree": {
            "name": "Decision Tree Classifier",
            "dir": "decision_tree",
            "pipeline_file": "decision_tree_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
        },
        "xgboost": {
            "name": "XGBoost Classifier",
            "dir": "xgboost",
            "pipeline_file": "xgboost_tuned_pipeline.joblib",
            "dropped_cols_file": "correlation_dropped_columns.joblib",
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
        self.model_type = model_type

        # In-memory artifact caches
        self._target_encoder = None
        self._scaler = None
        self._dropped_columns: Optional[List[str]] = None
        self._active_pipeline = None
        self._active_model_type: Optional[str] = None
        self._median_baseline: Optional[pd.Series] = None
        self._raw_feature_names: Optional[List[str]] = None
        self._test_X: Optional[pd.DataFrame] = None
        self._test_y: Optional[pd.DataFrame] = None

        # Pre-load core artifacts
        self.load_target_encoder()
        self.load_scaler()
        self.load_dropped_columns()
        self.load_model(model_type)

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
        if (parent_candidate / "final_dataset").exists():
            return parent_candidate

        # Check current working directory
        cwd_candidate = Path.cwd().resolve()
        if (cwd_candidate / "final_dataset").exists():
            return cwd_candidate

        # Check common Google Colab paths
        colab_candidate = Path("/content/Sports Analytics")
        if colab_candidate.exists():
            return colab_candidate

        colab_candidate_cwd = Path("/content")
        if (colab_candidate_cwd / "final_dataset").exists():
            return colab_candidate_cwd

        return parent_candidate

    # -------------------------------------------------------------------------
    # Artifact Loaders & Caching
    # -------------------------------------------------------------------------

    def load_target_encoder(self):
        """Load and cache the fitted target LabelEncoder."""
        if self._target_encoder is None:
            target_enc_path = self.base_dir / "final_dataset" / "target_encoder.joblib"
            if not target_enc_path.exists():
                raise FileNotFoundError(
                    f"Target encoder artifact missing at {target_enc_path}"
                )
            self._target_encoder = joblib.load(target_enc_path)
        return self._target_encoder

    def load_scaler(self):
        """Load and cache the fitted StandardScaler used for numerical normalization."""
        if self._scaler is None:
            scaler_path = self.base_dir / "final_dataset" / "scaler.joblib"
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
        return self._scaler

    def load_dropped_columns(self) -> List[str]:
        """Load and cache the 38 Stage-1 correlation dropped columns."""
        if self._dropped_columns is None:
            dropped_path = (
                self.base_dir
                / self.SUPPORTED_MODELS.get(self.model_type, {}).get(
                    "dir", "random_forest"
                )
                / "correlation_dropped_columns.joblib"
            )
            if not dropped_path.exists():
                dropped_path = (
                    self.base_dir / "random_forest" / "correlation_dropped_columns.joblib"
                )

            if not dropped_path.exists():
                raise FileNotFoundError(
                    f"Correlation dropped columns artifact not found at {dropped_path}"
                )
            self._dropped_columns = list(joblib.load(dropped_path))
        return self._dropped_columns

    def load_model(self, model_type: str = "random_forest"):
        """Dynamically load and cache the requested model pipeline artifact.

        Raises a clear, informative FileNotFoundError if the pipeline artifact is unpopulated.
        """
        if self._active_pipeline is not None and self._active_model_type == model_type:
            return self._active_pipeline

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Choose from {list(self.SUPPORTED_MODELS.keys())}"
            )

        model_info = self.SUPPORTED_MODELS[model_type]
        model_dir = self.base_dir / model_info["dir"]
        pipeline_path = model_dir / model_info["pipeline_file"]

        # Fallback: search for any .joblib file containing 'pipeline' in the model directory
        if not pipeline_path.exists() and model_dir.exists():
            joblib_candidates = list(model_dir.glob("*pipeline*.joblib"))
            if joblib_candidates:
                pipeline_path = joblib_candidates[0]

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Model pipeline artifact for '{model_info['name']}' not found at: {pipeline_path}\n"
                f"This model family is reserved for future training runs. Please select 'random_forest' "
                f"or train and place the pipeline artifact in '{pipeline_path}'."
            )

        self._active_pipeline = joblib.load(pipeline_path)
        self._active_model_type = model_type
        return self._active_pipeline

    def get_median_baseline(self) -> pd.Series:
        """Compute and cache in-memory the median feature baseline vector (114 features)."""
        if self._median_baseline is None:
            train_path = self.base_dir / "final_dataset" / "X_train_model.csv"
            if not train_path.exists():
                raise FileNotFoundError(f"Training dataset missing at {train_path}")

            df_train = pd.read_csv(train_path)
            self._raw_feature_names = list(df_train.columns)
            self._median_baseline = df_train.median()
        return self._median_baseline

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and cache the historical holdout test set (3923 fixtures)."""
        if self._test_X is None or self._test_y is None:
            test_x_path = self.base_dir / "final_dataset" / "X_test_model.csv"
            test_y_path = self.base_dir / "final_dataset" / "y_test.csv"

            if not test_x_path.exists() or not test_y_path.exists():
                raise FileNotFoundError(
                    f"Test dataset files missing at {test_x_path} or {test_y_path}"
                )

            self._test_X = pd.read_csv(test_x_path)
            self._test_y = pd.read_csv(test_y_path)

        return self._test_X, self._test_y

    # -------------------------------------------------------------------------
    # Scaling Helpers
    # -------------------------------------------------------------------------

    def scale_single_feature(self, col: str, val: float) -> float:
        """Scale a human-readable feature value using the fitted StandardScaler parameters."""
        scaler = self.load_scaler()
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            if col in scaler.feature_names_in_:
                idx = list(scaler.feature_names_in_).index(col)
                scale = scaler.scale_[idx]
                if scale > 0:
                    return float((val - scaler.mean_[idx]) / scale)
        return float(val)

    def unscale_single_feature(self, col: str, val: float) -> float:
        """Unscale a standardized feature value back to human-interpretable units."""
        scaler = self.load_scaler()
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            if col in scaler.feature_names_in_:
                idx = list(scaler.feature_names_in_).index(col)
                return float(val * scaler.scale_[idx] + scaler.mean_[idx])
        return float(val)

    # -------------------------------------------------------------------------
    # Feature Synthesis & Preprocessing
    # -------------------------------------------------------------------------

    def synthesize_simulation_features(
        self, input_dict: Dict[str, Any]
    ) -> pd.DataFrame:
        """Synthesize a single fixture feature vector overlaying user inputs on the median baseline.

        Drops the 38 multicollinear columns to output the exact 76 features expected by
        the trained machine learning pipeline.
        """
        # Obtain baseline template (114 features in scaled space)
        baseline = self.get_median_baseline().copy()
        df_row = baseline.to_frame().T

        # 1. League Selection (One-hot encoding, binary unscaled)
        selected_league = input_dict.get("league")
        if selected_league:
            for clean_name, col_name in self.CLEAN_LEAGUE_MAP.items():
                if col_name in df_row.columns:
                    df_row[col_name] = 0.0

            target_col = self.CLEAN_LEAGUE_MAP.get(selected_league)
            if not target_col and selected_league.startswith("league_name_"):
                target_col = selected_league
            if target_col and target_col in df_row.columns:
                df_row[target_col] = 1.0

        # 2. Stage / Matchweek (scaled)
        if "stage" in input_dict:
            raw_stage = float(input_dict["stage"])
            df_row["stage"] = self.scale_single_feature("stage", raw_stage)

        # 3. Market Consensus Odds (Bet365 + Align Pinnacle Odds, scaled)
        b365h = input_dict.get("B365H")
        b365d = input_dict.get("B365D")
        b365a = input_dict.get("B365A")

        if b365h is not None:
            val_h = float(b365h)
            df_row["B365H"] = self.scale_single_feature("B365H", val_h)
            if "PSH" in df_row.columns:
                df_row["PSH"] = self.scale_single_feature("PSH", float(input_dict.get("PSH", val_h)))
        if b365d is not None:
            val_d = float(b365d)
            df_row["B365D"] = self.scale_single_feature("B365D", val_d)
            if "PSD" in df_row.columns:
                df_row["PSD"] = self.scale_single_feature("PSD", float(input_dict.get("PSD", val_d)))
            if "GBD" in df_row.columns:
                df_row["GBD"] = self.scale_single_feature("GBD", float(input_dict.get("GBD", val_d)))
        if b365a is not None:
            val_a = float(b365a)
            df_row["B365A"] = self.scale_single_feature("B365A", val_a)
            if "PSA" in df_row.columns:
                df_row["PSA"] = self.scale_single_feature("PSA", float(input_dict.get("PSA", val_a)))

        # 4. Relative Performance & Form Differentials (Secondary Analytical Lens, scaled)
        # Goal Differentials
        goal_diff = input_dict.get(
            "goals_for_difference",
            input_dict.get("goal_difference_per_match"),
        )
        if goal_diff is not None:
            g_diff = float(goal_diff)
            df_row["goals_for_difference"] = self.scale_single_feature("goals_for_difference", g_diff)
            df_row["goals_against_difference"] = self.scale_single_feature("goals_against_difference", -g_diff)
            if "recent_goals_for_difference" in df_row.columns:
                df_row["recent_goals_for_difference"] = self.scale_single_feature("recent_goals_for_difference", g_diff * 0.9)
            if "recent_goals_against_difference" in df_row.columns:
                df_row["recent_goals_against_difference"] = self.scale_single_feature("recent_goals_against_difference", -g_diff * 0.9)

        # Win Rate Differentials
        win_rate_diff = input_dict.get(
            "win_rate_difference",
            input_dict.get("recent_win_rate_difference"),
        )
        if win_rate_diff is not None:
            w_diff = float(win_rate_diff)
            df_row["win_rate_difference"] = self.scale_single_feature("win_rate_difference", w_diff)
            if "recent_win_rate_difference" in df_row.columns:
                df_row["recent_win_rate_difference"] = self.scale_single_feature("recent_win_rate_difference", w_diff)
            # Adjust individual team baseline rates consistently
            unscaled_home_wr = self.unscale_single_feature("home_win_rate", float(baseline.get("home_win_rate", 0.0)))
            df_row["home_win_rate"] = self.scale_single_feature("home_win_rate", np.clip(unscaled_home_wr + (w_diff / 2.0), 0.05, 0.95))
            df_row["away_win_rate"] = self.scale_single_feature("away_win_rate", np.clip(unscaled_home_wr - (w_diff / 2.0), 0.05, 0.95))

        # Points Per Match Differentials
        ppm_diff = input_dict.get("points_per_match_difference")
        if ppm_diff is not None:
            df_row["points_per_match_difference"] = self.scale_single_feature("points_per_match_difference", float(ppm_diff))

        # Recent 5-Match Points Differential
        recent_pts_diff = input_dict.get(
            "recent_points_difference",
            input_dict.get("recent_5_match_points_difference"),
        )
        if recent_pts_diff is not None:
            df_row["recent_points_difference"] = self.scale_single_feature("recent_points_difference", float(recent_pts_diff))

        # Drop the 38 Stage-1 multicollinear features
        dropped_cols = self.load_dropped_columns()
        df_reduced = df_row.drop(columns=dropped_cols, errors="ignore")

        # Validate feature order against pipeline's expected input schema
        pipeline = self.load_model(self.model_type)
        if hasattr(pipeline, "feature_names_in_"):
            expected_features = list(pipeline.feature_names_in_)
            df_reduced = df_reduced.reindex(columns=expected_features)

        return df_reduced

    # -------------------------------------------------------------------------
    # Inference Methods
    # -------------------------------------------------------------------------

    def predict_simulation(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform match outcome prediction on a user-synthesized what-if simulation.

        Returns:
            Dict containing predicted_class_id, predicted_label, confidence,
            and probability distribution across all 3 outcome classes.
        """
        pipeline = self.load_model(self.model_type)
        encoder = self.load_target_encoder()

        # Synthesize and reduce feature vector to 76 features
        X_feature = self.synthesize_simulation_features(input_dict)

        # Run inference
        pred_id = int(pipeline.predict(X_feature)[0])
        probabilities = pipeline.predict_proba(X_feature)[0]

        # Extract class labels
        class_labels = list(encoder.classes_)  # ['Away Win', 'Draw', 'Home Win']
        pred_label = str(encoder.inverse_transform([pred_id])[0])

        prob_dict = {
            label: float(probabilities[idx])
            for idx, label in enumerate(class_labels)
        }
        confidence = round(float(np.max(probabilities)) * 100, 2)

        return {
            "predicted_class_id": pred_id,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "features_used_count": X_feature.shape[1],
        }

    def predict_test_match(self, match_index: int) -> Dict[str, Any]:
        """Evaluate a historical match fixture from the holdout test set (indices 0 to 3922).

        Returns prediction, probability distribution, ground-truth outcome,
        classification correctness status, and unscaled fixture metadata.
        """
        test_X, test_y = self.get_test_data()
        total_matches = len(test_X)

        if not (0 <= match_index < total_matches):
            raise IndexError(
                f"Test match index {match_index} out of bounds (0 to {total_matches - 1})."
            )

        pipeline = self.load_model(self.model_type)
        encoder = self.load_target_encoder()
        dropped_cols = self.load_dropped_columns()

        # Extract raw test fixture (114 features in scaled space)
        raw_row = test_X.iloc[[match_index]]

        # Extract fixture metadata with human-readable unscaled values
        metadata = self._extract_fixture_metadata(raw_row)

        # Drop collinear features (76 features)
        X_fixture = raw_row.drop(columns=dropped_cols, errors="ignore")
        if hasattr(pipeline, "feature_names_in_"):
            X_fixture = X_fixture.reindex(columns=list(pipeline.feature_names_in_))

        # Run prediction
        pred_id = int(pipeline.predict(X_fixture)[0])
        probabilities = pipeline.predict_proba(X_fixture)[0]
        pred_label = str(encoder.inverse_transform([pred_id])[0])

        # Ground truth
        actual_id = int(test_y.iloc[match_index, 0])
        actual_label = str(encoder.inverse_transform([actual_id])[0])
        is_correct = bool(pred_id == actual_id)

        class_labels = list(encoder.classes_)
        prob_dict = {
            label: float(probabilities[idx])
            for idx, label in enumerate(class_labels)
        }
        confidence = round(float(np.max(probabilities)) * 100, 2)

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
        }

    def _extract_fixture_metadata(self, row: pd.DataFrame) -> Dict[str, Any]:
        """Extract human-readable, unscaled metadata from a test match row."""
        league_detected = "European League"
        for clean_name, col in self.CLEAN_LEAGUE_MAP.items():
            if col in row.columns and float(row[col].values[0]) == 1.0:
                league_detected = clean_name
                break

        # Unscale stage and odds for human inspection
        raw_stage = row["stage"].values[0] if "stage" in row.columns else 1
        unscaled_stage = int(round(self.unscale_single_feature("stage", float(raw_stage))))

        b365h = self.unscale_single_feature("B365H", float(row["B365H"].values[0])) if "B365H" in row.columns else None
        b365d = self.unscale_single_feature("B365D", float(row["B365D"].values[0])) if "B365D" in row.columns else None
        b365a = self.unscale_single_feature("B365A", float(row["B365A"].values[0])) if "B365A" in row.columns else None

        return {
            "league": league_detected,
            "stage": max(1, min(38, unscaled_stage)),
            "odds": {
                "Home Win (B365H)": round(b365h, 2) if b365h else None,
                "Draw (B365D)": round(b365d, 2) if b365d else None,
                "Away Win (B365A)": round(b365a, 2) if b365a else None,
            },
        }
