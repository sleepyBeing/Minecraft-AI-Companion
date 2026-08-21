"""Deterministic, fixed-scenario evaluation for Stage Seven."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_7_prioritization import StageSevenPrioritizationEnv


def evaluate_policy(
    *,
    tf: Any,
    actor: Any,
    environment: StageSevenPrioritizationEnv,
    episode_count: int,
    base_seed: int,
) -> dict[str, float]:
    totals = {
        "return": 0.0,
        "successes": 0.0,
        "npc_deaths": 0.0,
        "bot_deaths": 0.0,
        "npc_damage": 0.0,
        "neutralizations": 0.0,
        "priority_neutralizations": 0.0,
        "wrong_priority_neutralizations": 0.0,
        "selector_actions": 0.0,
        "correct_target_steps": 0.0,
        "decisions": 0.0,
    }
    for episode in range(episode_count):
        observation, _ = environment.reset(seed=base_seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            logits = actor(
                tf.convert_to_tensor(observation[None, :], dtype=tf.float32),
                training=False,
            )
            action = int(tf.argmax(logits[0]).numpy())
            observation, reward, terminated, truncated, info = environment.step(action)
            totals["return"] += float(reward)
            totals["npc_damage"] += float(info["npc_damage_taken"])
            totals["neutralizations"] += float(int(info["neutralized_enemy"]) >= 0)
            totals["priority_neutralizations"] += float(
                info["neutralized_highest_priority"]
            )
            totals["wrong_priority_neutralizations"] += float(
                int(info["neutralized_enemy"]) >= 0
                and not info["neutralized_highest_priority"]
            )
            totals["selector_actions"] += float(action >= 12)
            totals["correct_target_steps"] += float(
                int(info["selected_enemy"]) == int(info["highest_priority_enemy"])
            )
            totals["decisions"] += 1.0
        totals["successes"] += float(info["success"])
        totals["npc_deaths"] += float(info["npc_defeated"])
        totals["bot_deaths"] += float(info["bot_defeated"])

    decisions = max(1.0, totals["decisions"])
    episodes = float(episode_count)
    return {
        "mean_return": totals["return"] / episodes,
        "success_rate": totals["successes"] / episodes,
        "npc_death_rate": totals["npc_deaths"] / episodes,
        "bot_death_rate": totals["bot_deaths"] / episodes,
        "mean_npc_damage": totals["npc_damage"] / episodes,
        "mean_neutralizations": totals["neutralizations"] / episodes,
        "mean_priority_neutralizations": (
            totals["priority_neutralizations"] / episodes
        ),
        "mean_wrong_priority_neutralizations": (
            totals["wrong_priority_neutralizations"] / episodes
        ),
        "correct_target_fraction": totals["correct_target_steps"] / decisions,
        "selector_action_fraction": totals["selector_actions"] / decisions,
        "mean_decisions": totals["decisions"] / episodes,
    }


def write_evaluation_summaries(
    *, tf: Any, summary_writer: Any, metrics: dict[str, float], step: int
) -> None:
    with summary_writer.as_default():
        for name, value in metrics.items():
            tf.summary.scalar(f"evaluation/{name}", value, step=step)
    summary_writer.flush()
    print(
        f"evaluation step={step:,} reward={metrics['mean_return']:+.2f} "
        f"success={metrics['success_rate']:.1%} "
        f"npc_death={metrics['npc_death_rate']:.1%} "
        f"bot_death={metrics['bot_death_rate']:.1%} "
        f"correct_target={metrics['correct_target_fraction']:.1%}"
    )
