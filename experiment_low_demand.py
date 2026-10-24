
"""Test whether normal-demand Q-Learning policies generalize to high demand.

Run from the project root:
    python experiment_demand.py

Agents train under normal demand and are evaluated under higher demand.
The evaluation price-change penalty is held constant across agents.
"""
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from src.environment import MarketConfig
from src.q_learning import train_agent
from src.evaluation import evaluate_policies, summarize_evaluation


RESULTS = Path("results") / "low_demand_experiment"

PENALTIES = (250.0, 750.0, 1500.0)
TRAINING_SEEDS = tuple(range(101, 121))
TRAINING_EPISODES = 1200
EVALUATION_EPISODES = 50
EVALUATION_SEED = 60000
HORIZON = 60

NORMAL_BASE_DEMAND = 22.0
LOW_BASE_DEMAND = 17.0
EVALUATION_PENALTY = 750.0

T_CRITICAL_20 = 2.093024054

METRICS = [
    "mean_reward",
    "mean_profit_before_penalties",
    "mean_units_sold",
    "mean_stockout_units",
    "mean_price",
    "mean_price_changes",
]


def mean_ci(values):
    values = pd.Series(values, dtype=float).dropna()
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    half_width = (
        T_CRITICAL_20 * sd / np.sqrt(n) if n > 1 else float("nan")
    )
    return {
        "n_training_runs": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    episode_frames = []
    run_summaries = []

    for penalty in PENALTIES:
        print(
            f"\n{'=' * 72}\n"
            f"Training penalty: ₹{penalty:.0f}\n"
            f"Training demand: {NORMAL_BASE_DEMAND:.0f}\n"
            f"Evaluation demand: {LOW_BASE_DEMAND:.0f}\n"
            f"{'=' * 72}"
        )

        for training_seed in TRAINING_SEEDS:
            print(
                f"Training seed {training_seed}: "
                f"{TRAINING_EPISODES} episodes..."
            )

            # Train in the original, normal-demand market.
            train_config = replace(
                MarketConfig(),
                base_demand=NORMAL_BASE_DEMAND,
                price_change_penalty=penalty,
                horizon=HORIZON,
                seed=training_seed,
            )

            agent, history = train_agent(
                episodes=TRAINING_EPISODES,
                horizon=HORIZON,
                seed=training_seed,
                config=train_config,
            )

            history_df = pd.DataFrame(history)
            history_df["training_penalty"] = penalty
            history_df["training_seed"] = training_seed
            history_df.to_csv(
                RESULTS
                / f"history_p{int(penalty)}_s{training_seed}.csv",
                index=False,
            )

            # Evaluate under lower demand with a common scoring penalty.
            evaluation_config = replace(
                MarketConfig(),
                base_demand=LOW_BASE_DEMAND,
                price_change_penalty=EVALUATION_PENALTY,
                horizon=HORIZON,
                seed=None,
            )

            episode_results, details = evaluate_policies(
                agent,
                episodes=EVALUATION_EPISODES,
                seed=EVALUATION_SEED,
                config=evaluation_config,
            )

            episode_results["training_penalty"] = penalty
            episode_results["training_seed"] = training_seed
            episode_frames.append(episode_results)

            details["training_penalty"] = penalty
            details["training_seed"] = training_seed
            details.to_csv(
                RESULTS
                / f"details_p{int(penalty)}_s{training_seed}.csv",
                index=False,
            )

            summary = summarize_evaluation(episode_results)
            summary["training_penalty"] = penalty
            summary["training_seed"] = training_seed
            run_summaries.append(summary)

            q = summary.loc[summary["policy"] == "Q-Learning"].iloc[0]
            print(
                f"  profit={q.mean_profit_before_penalties:.2f}; "
                f"stockouts={q.mean_stockout_units:.2f}; "
                f"price changes={q.mean_price_changes:.2f}"
            )

    episodes = pd.concat(episode_frames, ignore_index=True)
    per_seed = pd.concat(run_summaries, ignore_index=True)

    episodes.to_csv(RESULTS / "episode_results.csv", index=False)
    per_seed.to_csv(RESULTS / "per_seed_summary.csv", index=False)

    # Summarize the 20 independent training runs for every policy.
    aggregate_rows = []

    for (penalty, policy), group in per_seed.groupby(
        ["training_penalty", "policy"]
    ):
        row = {
            "training_penalty": penalty,
            "policy": policy,
        }

        for metric in METRICS:
            stats = mean_ci(group[metric])
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value

        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        RESULTS / "low_demand_summary_ci.csv", index=False
    )

    q_results = aggregate[aggregate["policy"] == "Q-Learning"]
    columns = [
        "training_penalty",
        "mean_profit_before_penalties_mean",
        "mean_profit_before_penalties_ci95_low",
        "mean_profit_before_penalties_ci95_high",
        "mean_stockout_units_mean",
        "mean_price_changes_mean",
    ]

    print("\nQ-Learning under low demand:")
    print(q_results[columns].round(2).to_string(index=False))

    print(f"\nSaved results to: {RESULTS.resolve()}")
    print(
        "Synthetic simulation only. Results do not establish "
        "real-world retail performance."
    )


if __name__ == "__main__":
    main()
