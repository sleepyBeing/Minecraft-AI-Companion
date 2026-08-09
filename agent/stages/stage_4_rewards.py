"""Small reward-shaping helpers for Stage Four."""

from __future__ import annotations


def bounded_positive_reward(
    raw_reward: float,
    earned_reward: float,
    reward_cap: float,
) -> tuple[float, float]:
    """Cap repeatable positive shaping without suppressing regress penalties."""
    if raw_reward <= 0.0:
        return raw_reward, earned_reward
    reward = min(raw_reward, max(0.0, reward_cap - earned_reward))
    return reward, earned_reward + reward
