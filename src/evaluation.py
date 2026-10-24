
"""Evaluate trained and baseline policies on held-out simulated episodes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.environment import DynamicPricingEnv, MarketConfig
from src.baselines import (
    fixed_price_policy,
    random_price_policy,
    rule_based_policy,
)


def run_episode(
    env,
    policy_name,
    agent=None,
    seed=10000,
    initial_price=None,
):
    """Run one episode and return day-level results.

    Q-Learning evaluation uses deterministic argmax tie-breaking.
    Training behavior is unchanged.
    """
    state = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 999)
    rows = []
    done = False

    if initial_price is None:
        initial_price = MarketConfig().reference_price

    while not done:
        if policy_name == "Q-Learning":
            state_index = env.encode_state(state)
            # np.argmax selects the first maximum when values are tied.
            action = int(np.argmax(agent.q_table[state_index]))
        elif policy_name == "Fixed price":
            action = fixed_price_policy(state, env)
        elif policy_name == "Rule-based":
            action = rule_based_policy(state, env)
        elif policy_name == "Random":
            action = random_price_policy(state, env, rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        state, reward, done, info = env.step(action)
        rows.append(info)

    return pd.DataFrame(rows)


def evaluate_policies(
    agent,
    episodes=50,
    seed=50000,
    config: MarketConfig | None = None,
):
    """Evaluate all policies on the same held-out episode seeds."""
    policy_names = ["Q-Learning", "Fixed price", "Rule-based", "Random"]
    evaluation_config = config or MarketConfig()
    initial_price = evaluation_config.reference_price

    summary_rows = []
    detail_frames = []

    for policy_name in policy_names:
        for episode in range(episodes):
            episode_seed = seed + episode

            # Each policy receives the same environment configuration and
            # episode seed, enabling a common-scenario comparison.
            from dataclasses import replace

            env_config = replace(evaluation_config, seed=None)
            env = DynamicPricingEnv(env_config)

            details = run_episode(
                env,
                policy_name,
                agent=agent,
                seed=episode_seed,
                initial_price=initial_price,
            )

            details["policy"] = policy_name
            details["episode"] = episode + 1
            detail_frames.append(details)

            prices = details["price"].to_numpy(dtype=float)
            previous_prices = np.concatenate(
                ([float(initial_price)], prices[:-1])
            )
            price_changes = int(
                np.count_nonzero(~np.isclose(prices, previous_prices))
            )

            summary_rows.append({
                "policy": policy_name,
                "episode": episode + 1,
                "total_reward": float(details["reward"].sum()),
                "total_profit": float(
                    details["profit_before_penalties"].sum()
                ),
                "total_units_sold": float(details["units_sold"].sum()),
                "total_revenue": float(details["revenue"].sum()),
                "stockout_units": float(details["unmet_demand"].sum()),
                "mean_price": float(details["price"].mean()),
                "price_changes": price_changes,
            })

    return (
        pd.DataFrame(summary_rows),
        pd.concat(detail_frames, ignore_index=True),
    )


def summarize_evaluation(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize episode outcomes by policy."""
    return (
        results.groupby("policy", as_index=False)
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_profit_before_penalties=("total_profit", "mean"),
            mean_units_sold=("total_units_sold", "mean"),
            mean_stockout_units=("stockout_units", "mean"),
            mean_price=("mean_price", "mean"),
            mean_price_changes=("price_changes", "mean"),
        )
        .sort_values("mean_reward", ascending=False)
    )
