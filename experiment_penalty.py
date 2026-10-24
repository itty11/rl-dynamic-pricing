"""Robust controlled experiment for price-change penalties.

Run from the project root: python experiment_penalty.py

Design: 3 penalties x 20 training seeds x 1,200 episodes, each agent evaluated
on the same 50 held-out market scenario seeds. Outputs include per-seed metrics,
95% t confidence intervals, and paired comparisons between penalty settings.
"""
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from src.environment import MarketConfig
from src.q_learning import train_agent
from src.evaluation import evaluate_policies, summarize_evaluation

RESULTS = Path("results") / "penalty_experiment_robust"
PENALTIES = (250.0, 750.0, 1500.0)
TRAINING_SEEDS = tuple(range(101, 121))  # 20 independent training runs per setting
TRAINING_EPISODES = 1200
EVALUATION_EPISODES = 50
EVALUATION_SEED = 50000
EVALUATION_PENALTY = 750.0
HORIZON = 60
# Two-sided 95% Student-t critical value for df=19 (20 training runs).
T_CRITICAL_20 = 2.093024054
METRICS = [
    "mean_reward", "mean_profit_before_penalties", "mean_units_sold",
    "mean_stockout_units", "mean_price", "mean_price_changes",
]


def mean_ci(values, t_critical=T_CRITICAL_20):
    values = pd.Series(values, dtype=float).dropna()
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    half_width = t_critical * sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n_training_runs": n, "mean": mean, "sd": sd,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    episode_frames, run_summaries = [], []

    for penalty in PENALTIES:
        print(f"\n{'=' * 72}\nPenalty: ₹{penalty:.0f} per 10% move\n{'=' * 72}")
        for training_seed in TRAINING_SEEDS:
            print(f"Training seed {training_seed}: {TRAINING_EPISODES} episodes...")
            # The only changed MarketConfig field is the price-change penalty.
            config = replace(MarketConfig(), price_change_penalty=penalty,
                             horizon=HORIZON, seed=training_seed)
            agent, history = train_agent(episodes=TRAINING_EPISODES,
                                         horizon=HORIZON, seed=training_seed,
                                         config=config)
            hist = pd.DataFrame(history)
            hist["penalty"] = penalty
            hist["training_seed"] = training_seed
            hist.to_csv(RESULTS / f"training_history_p{int(penalty)}_s{training_seed}.csv", index=False)

            # evaluate_policies derives episode seeds from EVALUATION_SEED and
            # resets each environment for each episode, common scenarios across runs.
            eval_config = replace(config, seed=None, price_change_penalty=EVALUATION_PENALTY,)
            episode_results, _details = evaluate_policies(
                agent, episodes=EVALUATION_EPISODES,
                seed=EVALUATION_SEED, config=eval_config)
            episode_results["penalty"] = penalty
            episode_results["training_seed"] = training_seed
            episode_frames.append(episode_results)

            summary = summarize_evaluation(episode_results)
            summary["penalty"] = penalty
            summary["training_seed"] = training_seed
            run_summaries.append(summary)
            q = summary.loc[summary.policy == "Q-Learning"].iloc[0]
            print(f"  reward={q.mean_reward:.2f}; stockouts={q.mean_stockout_units:.2f}; "
                  f"price changes={q.mean_price_changes:.2f}")

    episodes = pd.concat(episode_frames, ignore_index=True)
    per_seed = pd.concat(run_summaries, ignore_index=True)
    episodes.to_csv(RESULTS / "episode_results.csv", index=False)
    per_seed.to_csv(RESULTS / "per_seed_summary.csv", index=False)

    q_runs = per_seed[per_seed.policy == "Q-Learning"].copy()
    summary_rows = []
    for penalty, group in q_runs.groupby("penalty"):
        row = {"penalty": penalty}
        for metric in METRICS:
            stats = mean_ci(group[metric])
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        summary_rows.append(row)
    ci_summary = pd.DataFrame(summary_rows).sort_values("penalty")
    ci_summary.to_csv(RESULTS / "q_learning_penalty_ci_summary.csv", index=False)

    # Paired comparisons use matching training seed labels and the same held-out
    # evaluation scenarios. The CI is across the 20 per-seed metric differences.
    paired_rows = []
    for metric in METRICS:
        wide = q_runs.pivot(index="training_seed", columns="penalty", values=metric)
        for a, b in ((250.0, 750.0), (250.0, 1500.0), (750.0, 1500.0)):
            diff = wide[a] - wide[b]
            stats = mean_ci(diff)
            paired_rows.append({"metric": metric, "comparison": f"{int(a)} minus {int(b)}",
                                "penalty_a": a, "penalty_b": b,
                                "mean_difference_a_minus_b": stats["mean"],
                                "sd_difference": stats["sd"],
                                "ci95_low": stats["ci95_low"],
                                "ci95_high": stats["ci95_high"],
                                "n_training_seed_pairs": stats["n_training_runs"]})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(RESULTS / "paired_penalty_differences_ci.csv", index=False)

    all_policy = per_seed.groupby(["penalty", "policy"])[METRICS].agg(["mean", "std"])
    all_policy.columns = [f"{metric}_{stat}" for metric, stat in all_policy.columns]
    all_policy.reset_index().to_csv(RESULTS / "all_policies_summary.csv", index=False)

    print("\nQ-Learning mean and 95% CI across 20 training runs:")
    display_cols = ["penalty", "mean_reward_mean", "mean_reward_sd",
                    "mean_reward_ci95_low", "mean_reward_ci95_high",
                    "mean_stockout_units_mean", "mean_price_changes_mean"]
    print(ci_summary[display_cols].round(2).to_string(index=False))
    print(f"\nSaved robust experiment outputs to: {RESULTS.resolve()}")
    print("Synthetic simulator only; no real-world retail performance claim is established.")


if __name__ == "__main__":
    main()
