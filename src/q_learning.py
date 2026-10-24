"""Tabular Q-Learning agent and training utilities."""
from __future__ import annotations

import numpy as np

from src.environment import DynamicPricingEnv, MarketConfig


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.12,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.995,
        seed: int | None = 42,
    ):
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state_index: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))
        # Randomly break ties to avoid a fixed preference at initialization.
        values = self.q_table[state_index]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def learn(self, state_index, action, reward, next_state_index, done):
        current = self.q_table[state_index, action]
        future = 0.0 if done else float(np.max(self.q_table[next_state_index]))
        target = reward + self.gamma * future
        self.q_table[state_index, action] = current + self.lr * (target - current)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(
    episodes: int = 1200,
    horizon: int = 60,
    seed: int = 42,
    progress_callback=None,
    config: MarketConfig | None = None,
):
    """Train Q-Learning and return agent plus one-row-per-episode metrics.

    Pass a MarketConfig to run controlled experiments with different reward settings.
    """
    env_config = config or MarketConfig()
    # Use a copy so changing the horizon cannot mutate a caller-supplied config.
    from dataclasses import replace
    env_config = replace(env_config, horizon=horizon)
    env = DynamicPricingEnv(env_config)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    history = []

    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        total_reward = 0.0
        total_sales = 0
        total_revenue = 0.0
        for _ in range(horizon):
            state_index = env.encode_state(state)
            action = agent.choose_action(state_index, explore=True)
            next_state, reward, done, info = env.step(action)
            next_index = env.encode_state(next_state)
            agent.learn(state_index, action, reward, next_index, done)
            state = next_state
            total_reward += reward
            total_sales += info["units_sold"]
            total_revenue += info["revenue"]
            if done:
                break

        agent.decay_exploration()
        history.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "total_sales": total_sales,
            "total_revenue": total_revenue,
            "epsilon": agent.epsilon,
        })
        if progress_callback and (episode + 1) % 50 == 0:
            progress_callback(episode + 1, episodes)

    return agent, history
