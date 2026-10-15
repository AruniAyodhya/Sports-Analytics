### 1. Why We Performed Separate, Model-Specific Feature Selection

In classical machine learning, forcing a single, uniform feature subset across diverse model families violates the **No Free Lunch Theorem** and ignores the differing **inductive biases** of linear versus tree-based models:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               Raw Preprocessed Features (93)                           │
│                                           │                                            │
│                      Drop Market Betting Probabilities (7)                             │
│                      (Anti-Leakage / Pure Sports Analytics)                            │
│                                           │                                            │
│                       68 Pipeline Pre-Match Candidate Features                         │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│    Logistic Regression (L1 SFM)      │  │        Tree-Based Ensembles (RF / XGB / DT)  │
│ • Assumes linear hyperplanes         │  │ • Non-linear step functions & splits         │
│ • Highly sensitive to collinearity   │  │ • Invariant to monotonic scaling             │
│ • Requires StandardScaler            │  │ • Captures high-order feature interactions   │
│ • L1 penalty forces weights to 0     │  │ • Correlation Filter (>0.90 drop, 19 cols)   │
│ ──────────────────────────────────── │  │ ──────────────────────────────────────────── │
│ Final Model Dimension: 18 Features   │  │ RF (MDI SFM): 34  | XGB (Gain): 50 | DT: 20  │
└──────────────────────────────────────┘  └──────────────────────────────────────────────┘
```

#### A. Logistic Regression (L1 Lasso SelectFromModel — 18 Features)
* **Mathematical Sensitivity to Multicollinearity**: Logistic Regression models log-odds as a linear combination $\ln\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b$. When predictors are strongly correlated (e.g., `home_win_rate` and `home_points_per_match` with Pearson $r > 0.90$), the Hessian matrix $(\mathbf{X}^T \mathbf{X})$ becomes near-singular. This causes coefficient variance to explode, signs to flip erratically, and probabilities to become miscalibrated.
* **L1 Regularization Penalty**: By utilizing an $\ell_1$-norm penalty ($\lambda \sum |w_i|$), the diamond-shaped geometry of the parameter space forces redundant and noisy feature weights strictly to zero. 
* **StandardScaler Requirement**: Features must be standardized to zero mean and unit variance so penalty strength applies equally across scales.
* **Result**: A compact, non-collinear basis of **18 features** that preserves numerical stability and prevents overfitting on linear decision boundaries.

#### B. Tree Ensembles (Correlation Dropping + Non-Linear Selectors — 20 to 50 Features)
* **Non-Linear Interactions & Threshold Splitting**: Trees partition space using orthogonal threshold cuts ($\mathbb{I}(x_j \le \theta)$). They are invariant to monotonic scale and naturally model complex logic (e.g., *"if Elo difference > 150 AND rest days > 5"*).
* **Why Drop Correlations First (19 Columns)?**: In tree ensembles, if two features are 95%+ identical, Random Forest splits arbitrarily between them across bootstrapped samples, which artificially dilutes Mean Decrease in Impurity (MDI) across duplicate features. Pre-filtering pairs with $|r| > 0.90$ eliminates redundancy before tree construction.
* **Model-Specific Pruning**:
  * **Decision Tree (RFE — 20 Features)**: A single tree lacks ensemble bagging. Pruning down to 20 features via Recursive Feature Elimination prevents deep, high-variance leaf memorization.
  * **Random Forest (MDI SelectFromModel — 34 Features)**: Random Forest uses bagging. Filtering to the top 34 features retains the most informative splitters while maintaining tree diversity across 150 trees.
  * **XGBoost (Gain SelectFromModel — 50 Features)**: XGBoost uses gradient boosting with second-order Taylor expansion and built-in regularization ($\gamma, \lambda, \alpha$, shrinkage $\eta=0.08$). Gain measures the exact reduction in multi-class log-loss across sequential trees. It can safely exploit a richer space of **50 features** without overfitting.

---

### 2. How the 4 Core Differentials Interact with the 68-Feature Vector During Inference

In `views/simulator.py`, the user interacts with 4 intuitive tactical sliders:
1. `points_per_match_difference` (Points/Match Difference)
2. `recent_points_difference` (Recent 5-Match Points Difference)
3. `goals_for_difference` (Goal Difference/Match)
4. `win_rate_difference` (Recent Win Rate Difference)

However, our trained pipelines require the complete **68-feature input schema**. If the simulator merely altered those 4 columns and left the other 64 columns static or zeroed, the vector would become **internally contradictory** (e.g., asserting a +2.0 Goal Differential while individual scoring and conceding rates remain baseline, or claiming high form while Elo ratings remain unchanged).

To prevent distribution shift, [`SoccerInferenceEngine.synthesize_simulation_features()`](file:///f:/SLIIT/ML%20Assignment/Sports%20Analytics/src/inference.py#L573-L756) acts as a **domain physics engine** that coherently propagates these 4 sliders across all 68 features:

```
                          ┌──────────────────────────────────────────────┐
                          │     4 User Sliders (Simulation Inputs)       │
                          │ • Points/Match Diff   • Recent 5-Match Diff  │
                          │ • Goal Diff/Match     • Win Rate Diff        │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │  Domain Propagation Engine (inference.py)    │
                          │  Overlay on League Empirical Median Baseline │
                          └──────────────────────┬───────────────────────┘
                                                 │
         ┌────────────────────────┬──────────────┴───────────┬────────────────────────┐
         ▼                        ▼                          ▼                        ▼
┌──────────────────┐    ┌──────────────────┐       ┌──────────────────┐     ┌──────────────────┐
│ Elo Differentials│    │  Scoring & Goals │       │ Win & Loss Rates │     │ Rolling Form     │
│ • elo_difference │    │ • goals_for_diff │       │ • win_rate_diff  │     │ • points_last10  │
│ • home_elo_before│    │ • goals_against_d│       │ • venue_win_rate │     │ • points_last5   │
│ • away_elo_before│    │ • home_gf/away_ga│       │ • home_loss_rate │     │ • points_last3   │
│ (Scaled by Stage)│    │ • venue_gf / ga  │       │ • away_loss_rate │     │ • goals_last5_ga │
└──────────────────┘    └──────────────────┘       └──────────────────┘     └──────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │   Full Synthesized Vector (68 Features)      │
                          └──────────────────────┬───────────────────────┘
                                                 │
         ┌───────────────────────────────────────┴────────────────────────────────────┐
         ▼                                                                            ▼
┌────────────────────────────────────────┐                   ┌────────────────────────────────────────┐
│     Logistic Regression Pipeline       │                   │       Tree Ensemble Pipelines          │
│ 1. Standardize via StandardScaler      │                   │ 1. Keep Unscaled Domain Values         │
│ 2. Drop 19 collinear columns           │                   │ 2. Drop 19 collinear columns           │
│ 3. Pipeline L1 SFM filters to 18 feats │                   │ 3. Pipeline SFM/RFE filters to active  │
│ 4. Output Calibrated Probabilities     │                   │    features (RF: 34, XGB: 50, DT: 20)  │
└────────────────────────────────────────┘                   └────────────────────────────────────────┘
```

#### Mathematical Propagation Mechanics in `src/inference.py`:

1. **Empirical League Baseline Anchor**:
   Starts from the true historical median vector for the selected league (`get_league_baseline()`), establishing realistic base scoring paces ($1.30$ goals/match) and base draw rates ($0.26$).
2. **Season Maturity Modulation**:
   Differentials are modulated by season progression:
   $$\text{stage\_factor} = 0.75 + 0.50 \times \left(\frac{\text{stage}}{38}\right)$$
   *(Differentials early in Matchweek 2 carry less polarization than late in Matchweek 35).*
3. **Elo Rating Synthesis**:
   `win_rate_diff`, `goals_diff`, and `ppm_diff` synthesize the latent team strength:
   $$\text{elo\_diff} = \left(\text{offset} + 250 \times \Delta_{\text{WR}} + 90 \times \Delta_{\text{GD}} + 70 \times \Delta_{\text{PPM}}\right) \times \text{stage\_factor}$$
   Directly populating `elo_difference`, `home_elo_before`, and `away_elo_before`.
4. **Scoring and Conceding Decomposition**:
   `goals_diff` dynamically splits into individual attacking and defending rates:
   * $\text{home\_goals\_for\_per\_match} = \text{base} + 0.45 \times \Delta_{\text{GD}}$
   * $\text{away\_goals\_against\_per\_match} = \text{base} + 0.45 \times \Delta_{\text{GD}}$
   * Sets rolling 10-match, 5-match, and venue-specific scoring expectations.
5. **Short-Term Goal Conceding (Crucial for Logistic Regression)**:
   $$\text{goals\_against\_last5\_difference} = -0.4 \times \Delta_{\text{pts5}} - 0.4 \times \Delta_{\text{GD}}$$
   *(Logistic Regression's L1 selector identified this exact column as one of its top predictive weights).*
6. **Pipeline Schema Alignment & Transformation**:
   * **Logistic Regression**: Runs `StandardScaler.transform()`, drops correlation columns, and aligns with `feature_names_in_` where the L1 transformer extracts the 18 active features.
   * **Random Forest & XGBoost**: Retains natural unscaled values, drops correlation columns, and lets the pipeline's internal `SelectFromModel` isolate the exact 34 or 50 features.

This architecture ensures that regardless of which model is selected in the sidebar, user interactions on the 4 core sliders translate into **statistically grounded, leakage-free, and internally consistent feature vectors** across all 4 machine learning pipelines.
