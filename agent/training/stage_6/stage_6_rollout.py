"""Duration-aware rollout collection for Stage Six."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_6_protection import StageSixProtectionEnv
from agent.training.stage_5.stage_5_episode import (
    transition_metrics as stage_five_transition_metrics,
)
from agent.training.stage_5.stage_5_rollout import (
    EXTRA_NAMES as STAGE_FIVE_EXTRA_NAMES,
    _finish_rollout,
)
from agent.training.stage_6.stage_6_episode import (
    PROTECTION_METRICS,
    StageSixEpisodeCsvLogger,
    StageSixEpisodeState,
    finish_episode,
    protection_transition_metrics,
)


PROTECTION_EXTRAS = (
    *PROTECTION_METRICS,
    "npc_alive_steps",
    "npc_survived_episode_flags",
    "interaction_distance",
    "bot_npc_distance",
    "enemy_npc_distance",
)


def collect_stage_six_rollout(
    *,
    tf: Any,
    actor: Any,
    critic: Any,
    environment: StageSixProtectionEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_state: StageSixEpisodeState,
    episode_logger: StageSixEpisodeCsvLogger,
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageSixEpisodeState]:
    core_names = (
        "observations",
        "actions",
        "rewards",
        "values",
        "log_probabilities",
        "next_values",
        "terminated_flags",
        "episode_done_flags",
    )
    buffers: dict[str, list[Any]] = {
        name: []
        for name in (*core_names, *STAGE_FIVE_EXTRA_NAMES, *PROTECTION_EXTRAS)
    }
    observation = initial_observation

    for _ in range(rollout_size):
        observation_tensor = tf.convert_to_tensor(
            observation[None, :], dtype=tf.float32
        )
        logits = actor(observation_tensor, training=False)
        value = float(critic(observation_tensor, training=False)[0, 0].numpy())
        action = int(tf.random.categorical(logits, 1)[0, 0].numpy())
        log_probability = float(tf.nn.log_softmax(logits)[0, action].numpy())

        next_observation, reward, terminated, truncated, info = (
            environment.step(action)
        )
        next_value = float(
            critic(
                tf.convert_to_tensor(
                    next_observation[None, :], dtype=tf.float32
                ),
                training=False,
            )[0, 0].numpy()
        )
        episode_done = terminated or truncated
        stage_five = stage_five_transition_metrics(action, info)
        protection = protection_transition_metrics(info)
        values = {
            "observations": observation.copy(),
            "actions": action,
            "rewards": reward,
            "values": value,
            "log_probabilities": log_probability,
            "next_values": next_value,
            "terminated_flags": terminated,
            "episode_done_flags": episode_done,
            **stage_five,
            "action_duration_steps": float(info["action_duration_steps"]),
            "cover_occluded_steps": float(info["cover_occluded"]),
            "safe_to_eat_steps": float(info["safe_to_eat"]),
            "has_recovered_steps": float(info["has_recovered"]),
            "successes": float(info["success"]),
            **protection,
            "npc_alive_steps": float(info["npc_alive"]),
            "npc_survived_episode_flags": float(
                info["npc_survived_episode"]
            ),
            "interaction_distance": float(info["interaction_distance"]),
            "bot_npc_distance": float(info["bot_npc_distance"]),
            "enemy_npc_distance": float(info["enemy_npc_distance"]),
        }
        for name, value_to_append in values.items():
            buffers[name].append(value_to_append)

        episode_state.update(action, reward, info)
        observation = next_observation
        if episode_done:
            finish_episode(
                tf=tf,
                info=info,
                state=episode_state,
                logger=episode_logger,
                summary_writer=summary_writer,
            )
            observation, _ = environment.reset()
            episode_state.reset_episode()

    rollout = _finish_rollout(
        buffers=buffers,
        rollout_size=rollout_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    for name in PROTECTION_EXTRAS:
        rollout[name] = np.asarray(buffers[name], dtype=np.float32)
    return rollout, observation, episode_state
