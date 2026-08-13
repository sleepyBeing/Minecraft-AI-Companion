"""Duration-aware rollout collection for Stage Five."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_5_recovery import StageFiveRecoveryEnv
from agent.training.stage_5.stage_5_episode import (
    METRIC_NAMES,
    StageFiveEpisodeCsvLogger,
    StageFiveEpisodeState,
    finish_episode,
    transition_metrics,
)


EXTRA_NAMES = (
    *METRIC_NAMES,
    "action_duration_steps",
    "cover_occluded_steps",
    "safe_to_eat_steps",
    "has_recovered_steps",
    "successes",
)


def collect_stage_five_rollout(
    *,
    tf: Any,
    actor: Any,
    critic: Any,
    environment: StageFiveRecoveryEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_state: StageFiveEpisodeState,
    episode_logger: StageFiveEpisodeCsvLogger,
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageFiveEpisodeState]:
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
        name: [] for name in (*core_names, *EXTRA_NAMES)
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
        values = {
            "observations": observation.copy(),
            "actions": action,
            "rewards": reward,
            "values": value,
            "log_probabilities": log_probability,
            "next_values": next_value,
            "terminated_flags": terminated,
            "episode_done_flags": episode_done,
            **transition_metrics(action, info),
            "action_duration_steps": float(info["action_duration_steps"]),
            "cover_occluded_steps": float(info["cover_occluded"]),
            "safe_to_eat_steps": float(info["safe_to_eat"]),
            "has_recovered_steps": float(info["has_recovered"]),
            "successes": float(info["success"]),
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
    return rollout, observation, episode_state


def _finish_rollout(
    *,
    buffers: dict[str, list[Any]],
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
) -> dict[str, np.ndarray]:
    rewards = np.asarray(buffers["rewards"], dtype=np.float32)
    values = np.asarray(buffers["values"], dtype=np.float32)
    next_values = np.asarray(buffers["next_values"], dtype=np.float32)
    terminated = np.asarray(buffers["terminated_flags"], dtype=np.float32)
    episode_done = np.asarray(
        buffers["episode_done_flags"], dtype=np.float32
    )
    durations = np.asarray(
        buffers["action_duration_steps"], dtype=np.float32
    )
    discounts = np.power(gamma, durations).astype(np.float32)
    trace_discounts = np.power(gamma * gae_lambda, durations).astype(
        np.float32
    )
    deltas = rewards + discounts * next_values * (1.0 - terminated) - values

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for index in range(rollout_size - 1, -1, -1):
        gae = (
            deltas[index]
            + trace_discounts[index]
            * (1.0 - episode_done[index])
            * gae
        )
        advantages[index] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    rollout = {
        "observations": np.asarray(
            buffers["observations"], dtype=np.float32
        ),
        "actions": np.asarray(buffers["actions"], dtype=np.int32),
        "rewards": rewards,
        "values": values,
        "log_probabilities": np.asarray(
            buffers["log_probabilities"], dtype=np.float32
        ),
        "advantages": advantages,
        "returns": returns,
    }
    for name in EXTRA_NAMES:
        rollout[name] = np.asarray(buffers[name], dtype=np.float32)
    return rollout
