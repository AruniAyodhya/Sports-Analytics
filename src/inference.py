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
        if (parent_candidate / "final_dataset").exists() or (parent_candidate / "random_forest").exists():
            return parent_candidate

        # Check current working directory
        cwd_candidate = Path.cwd().resolve()
        if (cwd_candidate / "final_dataset").exists() or (cwd_candidate / "random_forest").exists():
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
            scaler_path = self.base_dir / "final_dataset" / "logistic_scaler.joblib"
            if not scaler_path.exists():
                scaler_path = self.base_dir / "final_dataset" / "scaler.joblib"
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
        return self._scaler

    def load_dropped_columns(self, model_type: Optional[str] = None) -> List[str]:
        """Load and cache the Stage-1 correlation dropped columns for a given model."""
        m_type = model_type or self.model_type
        if m_type in self._dropped_columns:
            return self._dropped_columns[m_type]

        model_info = self.SUPPORTED_MODELS.get(m_type, self.SUPPORTED_MODELS["random_forest"])
        model_dir = self.base_dir / model_info["dir"]

        dropped_path = model_dir / model_info.get("dropped_cols_file", "correlation_dropped_columns.joblib")
        if not dropped_path.exists():
            dropped_path = model_dir / "correlation_dropped_columns.joblib"
        if not dropped_path.exists():
            dropped_path = model_dir / "correlation_drop_columns.joblib"
        if not dropped_path.exists():
            dropped_path = self.base_dir / "random_forest" / "correlation_dropped_columns.joblib"

        if not dropped_path.exists():
            raise FileNotFoundError(
                f"Correlation dropped columns artifact not found at {dropped_path}"
            )
        self._dropped_columns[m_type] = list(joblib.load(dropped_path))
        return self._dropped_columns[m_type]

    def load_threshold(self, model_type: Optional[str] = None) -> float:
        """Load and cache the calibrated optimal draw threshold for a model (defaults to 0.360)."""
        m_type = model_type or self.model_type
        if m_type in self._thresholds:
            return self._thresholds[m_type]

        model_info = self.SUPPORTED_MODELS.get(m_type, self.SUPPORTED_MODELS["random_forest"])
        threshold_path = self.base_dir / model_info["dir"] / model_info.get("threshold_file", "calibrated_draw_threshold.joblib")

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

        Raises a clear, informative FileNotFoundError if the pipeline artifact is unpopulated.
        """
        if model_type in self._pipeline_cache:
            self.model_type = model_type
            return self._pipeline_cache[model_type]

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
                f"Please ensure models are trained and placed in their respective directories."
            )

        pipeline = joblib.load(pipeline_path)
        self._pipeline_cache[model_type] = pipeline
        self.model_type = model_type
        return pipeline

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
            tree_train_path = self.base_dir / "final_dataset" / "X_train_tree.csv"
            if not tree_train_path.exists():
                tree_train_path = self.base_dir / "final_dataset" / "X_train_model.csv"

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
        """Load and cache the historical holdout test set."""
        m_type = model_type or self.model_type
        family = self.SUPPORTED_MODELS.get(m_type, {}).get("family", "tree")

        if family not in self._test_X_cache or self._test_y is None:
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
            meta_path = self.base_dir / "final_dataset" / "test_metadata.csv"
            if meta_path.exists():
                self._test_metadata = pd.read_csv(meta_path)
            else:
                raw_split_path = self.base_dir / "splitted_dataset" / "test.csv"
                if raw_split_path.exists():
                    self._test_metadata = pd.read_csv(raw_split_path)
        return self._test_metadata

    # -------------------------------------------------------------------------
    # Scaling Helpers & Calibration
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

    @staticmethod
    def predict_with_draw_threshold(p: np.ndarray, threshold: float = 0.360) -> int:
        """Classify 3-class probability distribution with calibrated draw threshold.

        Class 0: Away Win
        Class 1: Draw
        Class 2: Home Win
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
        """Synthesize a complete fixture feature vector overlaying user inputs on the empirical baseline.

        Ensures every dynamic UI input (league, stage, market odds, points-per-match, recent points,
        goal differential, win rate differential) coherently propagates across the exact feature subsets
        utilized by Random Forest, Logistic Regression, Decision Tree, and XGBoost.
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

        # 4. User Sliders & Form Differentials (Pure Team Performance Metrics)
        ppm_diff = float(input_dict.get("points_per_match_difference", 0.5))
        recent_pts_diff = float(input_dict.get("recent_points_difference", 0.8))
        goals_diff = float(input_dict.get("goals_for_difference", input_dict.get("goal_difference_per_match", 0.4)))
        win_rate_diff = float(input_dict.get("win_rate_difference", input_dict.get("recent_win_rate_difference", 0.2)))

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
        """Extract human-readable, unscaled metadata from a test match row."""
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
