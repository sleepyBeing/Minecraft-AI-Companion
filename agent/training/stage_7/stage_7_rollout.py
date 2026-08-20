"""Duration-aware rollout collection for Stage Seven."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_7_prioritization import StageSevenPrioritizationEnv
from agent.training.stage_5.stage_5_rollout import (
    EXTRA_NAMES as STAGE_FIVE_EXTRAS,
    _finish_rollout,
)
from agent.training.stage_6.stage_6_rollout import PROTECTION_EXTRAS
from agent.training.stage_7.stage_7_episode import (
    PRIORITY_METRICS,
    StageSevenEpisodeCsvLogger,
    StageSevenEpisodeState,
    finish_episode,
    inherited_transition_metrics,
    priority_transition_metrics,
)


PRIORITY_EXTRAS = (
    *PRIORITY_METRICS,
    "enemy_1_alive_steps",
    "enemy_2_alive_steps",
    "enemy_1_health",
    "enemy_2_health",
)


def collect_stage_seven_rollout(
    *, tf: Any, actor: Any, critic: Any,
    environment: StageSevenPrioritizationEnv,
    initial_observation: np.ndarray, rollout_size: int,
    gamma: float, gae_lambda: float,
    episode_state: StageSevenEpisodeState,
    episode_logger: StageSevenEpisodeCsvLogger,
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageSevenEpisodeState]:
    core = (
        "observations", "actions", "rewards", "values", "log_probabilities",
        "next_values", "terminated_flags", "episode_done_flags",
    )
    buffers: dict[str, list[Any]] = {
        name: []
        for name in (*core, *STAGE_FIVE_EXTRAS, *PROTECTION_EXTRAS, *PRIORITY_EXTRAS)
    }
    observation = initial_observation

    for _ in range(rollout_size):
        tensor = tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
        logits = actor(tensor, training=False)
        value = float(critic(tensor, training=False)[0, 0].numpy())
        action = int(tf.random.categorical(logits, 1)[0, 0].numpy())
        log_probability = float(tf.nn.log_softmax(logits)[0, action].numpy())
        next_observation, reward, terminated, truncated, info = environment.step(action)
        next_value = float(critic(
            tf.convert_to_tensor(next_observation[None, :], dtype=tf.float32),
            training=False,
        )[0, 0].numpy())
        episode_done = terminated or truncated
        inherited = inherited_transition_metrics(action, info)
        priority = priority_transition_metrics(action, info)
        values = {
            "observations": observation.copy(),
            "actions": action,
            "rewards": reward,
            "values": value,
            "log_probabilities": log_probability,
            "next_values": next_value,
            "terminated_flags": terminated,
            "episode_done_flags": episode_done,
            **inherited,
            "action_duration_steps": float(info["action_duration_steps"]),
            "cover_occluded_steps": float(info["cover_occluded"]),
            "safe_to_eat_steps": float(info["safe_to_eat"]),
            "has_recovered_steps": float(info["has_recovered"]),
            "successes": float(info["success"]),
            "npc_alive_steps": float(info["npc_alive"]),
            "npc_survived_episode_flags": float(info["npc_survived_episode"]),
            "interaction_distance": float(info["interaction_distance"]),
            "bot_npc_distance": float(info["bot_npc_distance"]),
            "enemy_npc_distance": float(info["enemy_npc_distance"]),
            **priority,
            "enemy_1_alive_steps": float(info["enemy_alive"][0]),
            "enemy_2_alive_steps": float(info["enemy_alive"][1]),
            "enemy_1_health": float(info["enemy_health"][0]),
            "enemy_2_health": float(info["enemy_health"][1]),
        }
        for name, item in values.items():
            buffers[name].append(item)

        episode_state.update(action, reward, info)
        observation = next_observation
        if episode_done:
            finish_episode(
                tf=tf, info=info, state=episode_state,
                logger=episode_logger, summary_writer=summary_writer,
            )
            observation, _ = environment.reset()
            episode_state.reset_episode()

    rollout = _finish_rollout(
        buffers=buffers,
        rollout_size=rollout_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    for name in (*PROTECTION_EXTRAS, *PRIORITY_EXTRAS):
        rollout[name] = np.asarray(buffers[name], dtype=np.float32)
    return rollout, observation, episode_state
