# RL Dynamic Pricing — Consolidated Penalty Experiment Analysis

## Scope and data

This report consolidates the three uploaded per-seed result files:

- Low demand: base demand 17 (`per_seed_summary.csv`).
- Normal demand: base demand 22 (`per_seed_summary(2).csv`, robust penalty experiment).
- High demand: base demand 27 (`per_seed_summary(1).csv`).

Each condition contains 3 training penalty settings (₹250, ₹750, ₹1,500), 20 training seeds (101–120), and four policies. The combined per-seed file retains the original rows and adds demand labels and source-file metadata.

The robust normal-demand script uses 1,200 training episodes per agent, 50 evaluation episodes, evaluation seed 50,000, and a common evaluation penalty of ₹750. The low/high scripts were previously described as using evaluation seed 60,000. Therefore, cross-demand comparisons are descriptive rather than fully matched on identical evaluation scenarios.

## Q-Learning averages

| Demand (base demand) | Training penalty | Mean profit before penalties | Mean reward | Mean stockout units | Mean price changes |
|---|---:|---:|---:|---:|---:|
| Low (17) | ₹250 | 329,887.60 | 291,270.80 | 216.32 | 32.50 |
| Low (17) | ₹750 | 343,173.00 | 317,167.11 | 185.61 | 21.79 |
| Low (17) | ₹1,500 | 350,653.90 | 331,384.77 | 166.59 | 16.73 |
| Normal (22) | ₹250 | 336,012.50 | 291,341.40 | 497.91 | 31.71 |
| Normal (22) | ₹750 | 351,752.10 | 321,685.94 | 451.48 | 19.60 |
| Normal (22) | ₹1,500 | 360,335.40 | 337,969.30 | 422.41 | 13.71 |
| High (27) | ₹250 | 336,773.20 | 285,076.64 | 785.05 | 31.67 |
| High (27) | ₹750 | 353,402.20 | 316,825.58 | 723.31 | 19.42 |
| High (27) | ₹1,500 | 362,168.50 | 333,651.67 | 686.77 | 13.32 |

## Interpretation

1. Across all three demand conditions, the observed Q-Learning averages improve when the training penalty rises from ₹250 to ₹1,500: mean profit before penalties and mean reward rise, while mean stockout units and price changes fall.
2. The ₹750 setting generally improves on ₹250, and the paired 95% confidence intervals can be inspected in `paired_penalty_comparisons.csv`.
3. Although ₹1,500 has the strongest observed averages, the paired confidence intervals for ₹1,500 versus ₹750 should be used to judge uncertainty; do not claim it is definitively superior where intervals include zero.
4. Confidence intervals are Student-t intervals across 20 training-seed-level results. The paired intervals are unadjusted for multiple comparisons.
5. These are synthetic simulator outcomes, not proof of real-world retail performance.

## Metric definitions and caveats

- `mean_profit_before_penalties`: simulator profit metric before subtracting the explicit price-change penalty. It should not be confused with the total reward.
- `mean_reward`: reward under the common evaluation penalty setting (₹750 in the robust normal-demand experiment).
- `mean_stockout_units`: average unmet-demand/stockout units per evaluation episode, as defined by the project evaluator.
- `mean_price_changes`: average count of price changes per episode.
- The per-seed rows are the statistical units for confidence intervals; the 50 episodes used to evaluate each trained agent are already aggregated within each training seed.

## Recommended next research step

Before any new training, run a controlled evaluation-only comparison on a shared set of held-out scenario seeds for all three demand conditions and all trained agents, keeping evaluation penalty fixed at ₹750. This isolates the effect of demand condition from the different evaluation-seed sets. Retraining should not be needed if the trained agents/models are saved; if models were not saved, use the existing per-seed analysis and clearly state the scenario-seed limitation rather than immediately repeating all training.

## Files

- `combined_per_seed_results.csv`: all 720 per-seed policy rows.
- `q_learning_summary_with_95ci.csv`: Q-Learning means, standard deviations, and 95% t confidence intervals.
- `paired_penalty_comparisons.csv`: paired differences (higher penalty minus lower penalty), by metric and demand condition.
- Four PNG charts with means and 95% confidence intervals.
