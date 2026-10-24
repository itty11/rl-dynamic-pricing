"""Simple comparison policies for the simulated pricing market."""
from src.environment import DynamicPricingEnv


def fixed_price_policy(state, env):
    return 1  # reference price


def random_price_policy(state, env, rng):
    return int(rng.integers(env.n_actions))


def rule_based_policy(state, env):
    inventory_bin, demand_level, competitor_relation, previous_action = state

    # In high demand, increase price if stock is plentiful.
    if demand_level == 2 and inventory_bin >= 2:
        return 2
    # When demand is low, cut price if stock is available.
    if demand_level == 0 and inventory_bin >= 1:
        return 0
    # If the competitor is much cheaper, consider reducing our price.
    if competitor_relation == 0 and inventory_bin >= 1:
        return 0
    return 1
