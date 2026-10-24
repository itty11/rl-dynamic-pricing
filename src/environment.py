"""A small, self-contained simulated market for RL dynamic pricing.

All demand and competitor behavior are synthetic assumptions, not real market data.
The agent observes a coarse demand signal, inventory band, competitor price band,
and its previous price action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MarketConfig:
    reference_price: float = 1000.0
    unit_cost: float = 600.0
    initial_inventory: int = 60
    max_inventory: int = 80
    daily_replenishment: int = 12
    holding_cost_per_unit: float = 1.0
    # Opportunity-cost proxy for demand that could not be fulfilled.
    stockout_penalty_per_unit: float = 25.0
    horizon: int = 60
    base_demand: float = 22.0
    demand_noise_std: float = 3.0
    price_elasticity: float = 1.6
    competitor_influence: float = 0.25
    # Cost per 10-percentage-point price move (900↔1000 or 1000↔1100).
    price_change_penalty: float = 750.0
    seed: Optional[int] = None


class DynamicPricingEnv:
    """Minimal Gym-like environment; no Gymnasium dependency is required."""

    ACTION_MULTIPLIERS = (0.90, 1.00, 1.10)
    ACTION_NAMES = ("decrease", "maintain", "increase")

    # State: (inventory_bin, demand_level, competitor_relation, previous_action)
    # inventory_bin: 0 low, 1 medium, 2 high, 3 very high
    # demand_level: 0 low, 1 medium, 2 high
    # competitor_relation: 0 competitor cheaper, 1 similar, 2 competitor more expensive
    # previous_action: 0 decrease, 1 maintain, 2 increase

    def __init__(self, config: Optional[MarketConfig] = None):
        self.config = config or MarketConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.day = 0
        self.inventory = self.config.initial_inventory
        self.previous_action = 1
        self.last_info = {}
        self._current_demand_level = 1
        self._competitor_price = self.config.reference_price

    @property
    def n_actions(self) -> int:
        return len(self.ACTION_MULTIPLIERS)

    @property
    def n_states(self) -> int:
        return 4 * 3 * 3 * 3

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.day = 0
        self.inventory = self.config.initial_inventory
        self.previous_action = 1
        self._prepare_market_signal()
        return self._get_state()

    def _prepare_market_signal(self):
        # A seasonal demand pattern plus noise. The coarse signal is assumed observable.
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * self.day / 30.0)
        signal = seasonal + self.rng.normal(0, 0.18)
        if signal < 0.88:
            self._current_demand_level = 0
        elif signal < 1.12:
            self._current_demand_level = 1
        else:
            self._current_demand_level = 2

        # Competitor price varies independently around the reference price.
        self._competitor_price = self.config.reference_price * self.rng.choice(
            [0.90, 1.00, 1.10], p=[0.30, 0.40, 0.30]
        )

    def _get_state(self):
        inv_ratio = self.inventory / max(1, self.config.max_inventory)
        if inv_ratio < 0.20:
            inventory_bin = 0
        elif inv_ratio < 0.45:
            inventory_bin = 1
        elif inv_ratio < 0.75:
            inventory_bin = 2
        else:
            inventory_bin = 3

        ratio = self._competitor_price / self.config.reference_price
        if ratio < 0.95:
            competitor_relation = 0
        elif ratio > 1.05:
            competitor_relation = 2
        else:
            competitor_relation = 1

        return (
            inventory_bin,
            self._current_demand_level,
            competitor_relation,
            self.previous_action,
        )

    @staticmethod
    def encode_state(state) -> int:
        """Convert a 4-part discrete state to an integer Q-table index."""
        inventory_bin, demand_level, competitor_relation, previous_action = state
        return (((inventory_bin * 3 + demand_level) * 3 + competitor_relation) * 3
                + previous_action)

    def step(self, action: int):
        if action not in range(self.n_actions):
            raise ValueError(f"action must be 0, 1, or 2; received {action}")

        cfg = self.config
        old_inventory = self.inventory

        # Replenish before sales, capped by storage capacity.
        self.inventory = min(
            cfg.max_inventory, self.inventory + cfg.daily_replenishment
        )

        multiplier = self.ACTION_MULTIPLIERS[action]
        price = cfg.reference_price * multiplier

        # Expected demand responds to our price, competitor price, and demand signal.
        demand_level_multiplier = (0.70, 1.00, 1.35)[self._current_demand_level]
        relative_price = price / cfg.reference_price
        relative_competitor = self._competitor_price / cfg.reference_price

        price_effect = max(0.15, relative_price ** (-cfg.price_elasticity))
        competitor_effect = max(
            0.65, 1.0 + cfg.competitor_influence * (relative_competitor - relative_price)
        )
        expected_demand = (
            cfg.base_demand * demand_level_multiplier * price_effect * competitor_effect
        )
        demand = max(
            0, int(round(expected_demand + self.rng.normal(0, cfg.demand_noise_std)))
        )

        units_sold = min(demand, self.inventory)
        unmet_demand = max(0, demand - units_sold)
        ending_inventory = self.inventory - units_sold

        revenue = price * units_sold
        cost_of_goods = cfg.unit_cost * units_sold
        holding_cost = cfg.holding_cost_per_unit * ending_inventory
        stockout_penalty = cfg.stockout_penalty_per_unit * unmet_demand
        # Penalize the magnitude of a price move, not merely whether the
        # action changed. A 900→1100 move costs twice as much as 900→1000.
        previous_multiplier = self.ACTION_MULTIPLIERS[self.previous_action]
        price_move_steps = abs(multiplier - previous_multiplier) / 0.10
        change_penalty = cfg.price_change_penalty * price_move_steps
        reward = revenue - cost_of_goods - holding_cost - stockout_penalty - change_penalty

        self.inventory = ending_inventory
        self.previous_action = action
        self.day += 1

        self.last_info = {
            "day": self.day,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "price": price,
            "competitor_price": self._competitor_price,
            "demand_level": ("low", "medium", "high")[self._current_demand_level],
            "demand": demand,
            "units_sold": units_sold,
            "unmet_demand": unmet_demand,
            "ending_inventory": ending_inventory,
            "revenue": revenue,
            "profit_before_penalties": revenue - cost_of_goods,
            "reward": reward,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "price_change_penalty": change_penalty,
            "old_inventory": old_inventory,
        }

        done = self.day >= cfg.horizon
        if not done:
            self._prepare_market_signal()
        next_state = self._get_state()
        return next_state, float(reward), done, dict(self.last_info)
