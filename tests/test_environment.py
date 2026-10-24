from src.environment import DynamicPricingEnv, MarketConfig


def test_reset_and_step():
    env = DynamicPricingEnv(MarketConfig(horizon=3, seed=123))
    state = env.reset(seed=123)
    assert len(state) == 4
    next_state, reward, done, info = env.step(1)
    assert len(next_state) == 4
    assert isinstance(reward, float)
    assert "price" in info
    assert info["price"] == 1000.0
    assert done is False


def test_episode_ends_at_horizon():
    env = DynamicPricingEnv(MarketConfig(horizon=2, seed=1))
    env.reset(seed=1)
    _, _, done, _ = env.step(1)
    assert done is False
    _, _, done, _ = env.step(1)
    assert done is True


def test_invalid_action_raises():
    env = DynamicPricingEnv()
    env.reset(seed=1)
    try:
        env.step(9)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_price_change_penalty_scales_with_move_size():
    env = DynamicPricingEnv(MarketConfig(horizon=3, seed=7, price_change_penalty=250.0))
    env.reset(seed=7)
    # Reset starts at the reference-price action (1000).
    _, _, _, small_move = env.step(0)  # 1000 -> 900: one 10% step
    assert abs(small_move["price_change_penalty"] - 250.0) < 1e-8

    # From action 0 (900), action 2 jumps to 1100: two 10% steps.
    _, _, _, large_move = env.step(2)
    assert abs(large_move["price_change_penalty"] - 500.0) < 1e-8


def test_stockout_penalty_uses_configured_rate():
    cfg = MarketConfig(stockout_penalty_per_unit=25.0, seed=3)
    env = DynamicPricingEnv(cfg)
    env.reset(seed=3)
    _, _, _, info = env.step(0)
    assert info["stockout_penalty"] == info["unmet_demand"] * 25.0
