"""Train, evaluate, and save artifacts for the RL dynamic pricing project."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.q_learning import train_agent
from src.evaluation import evaluate_policies, summarize_evaluation

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def main():
    episodes = 1200
    print(f"Training Q-Learning for {episodes} episodes...")
    agent, history = train_agent(episodes=episodes)
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS / "training_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["episode"], history_df["total_reward"], alpha=0.25, label="Episode reward")
    plt.plot(
        history_df["episode"],
        history_df["total_reward"].rolling(50, min_periods=1).mean(),
        label="50-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total reward (₹, simulated)")
    plt.title("Q-Learning training progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "learning_curve.png", dpi=150)
    plt.close()

    print("Evaluating policies on held-out simulated episodes...")
    results, details = evaluate_policies(agent, episodes=50)
    summary = summarize_evaluation(results)
    results.to_csv(RESULTS / "evaluation_episodes.csv", index=False)
    details.to_csv(RESULTS / "evaluation_details.csv", index=False)
    summary.to_csv(RESULTS / "evaluation_summary.csv", index=False)
    pd.DataFrame(agent.q_table).to_csv(RESULTS / "q_table.csv", index=False)

    print("\nEvaluation summary (means over 50 episodes):")
    print(summary.round(2).to_string(index=False))
    print("\nSaved outputs to the results/ folder.")
    print("Note: all results are from a synthetic simulator, not real retail performance.")


if __name__ == "__main__":
    main()
