"""Rollout collection and episode logging for Stage Two."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from agent.stages.stage_2_stationary_combat import StageTwoStationaryCombatEnv


class StageTwoEpisodeState:
    def __init__(self) -> None:
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_number = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.bot_defeats = 0
        self.attack_selections = 0
        self.valid_attack_attempts = 0
        self.invalid_attacks = 0
        self.target_tracking_failures = 0
        self.movement_settle_failures = 0
        self.cooldown_blocked_attacks = 0
        self.confirmed_hits = 0
        self.range_entries = 0

    def reset_episode(self) -> None:
        self.episode_return = 0.0
        self.episode_length = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.bot_defeats = 0
        self.attack_selections = 0
        self.valid_attack_attempts = 0
        self.invalid_attacks = 0
        self.target_tracking_failures = 0
        self.movement_settle_failures = 0
        self.cooldown_blocked_attacks = 0
        self.confirmed_hits = 0
        self.range_entries = 0


def collect_stage_two_rollout(
    *,
    tf: Any,
    actor: Any,
    critic: Any,
    environment: StageTwoStationaryCombatEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_state: StageTwoEpisodeState,
    episode_logger: "StageTwoEpisodeCsvLogger",
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageTwoEpisodeState]:
    buffers: dict[str, list[Any]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "log_probabilities": [],
        "next_values": [],
        "terminated_flags": [],
        "episode_done_flags": [],
        "damage_dealt": [],
        "damage_taken": [],
        "bot_defeats": [],
        "approach_rewards": [],
        "attack_selections": [],
        "valid_attack_attempts": [],
        "invalid_attacks": [],
        "target_tracking_failures": [],
        "movement_settle_failures": [],
        "cooldown_blocked_attacks": [],
        "confirmed_hits": [],
    }
    observation = initial_observation

    for _ in range(rollout_size):
        observation_tensor = tf.convert_to_tensor(
            observation[None, :], dtype=tf.float32
        )
        logits = actor(observation_tensor, training=False)
        value = float(critic(observation_tensor, training=False)[0, 0].numpy())
        action = int(tf.random.categorical(logits, 1)[0, 0].numpy())
        log_probability = float(
            tf.nn.log_softmax(logits)[0, action].numpy()
        )

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
        damage = float(info["damage_dealt"])
        damage_taken = float(info.get("damage_taken", 0.0))
        bot_defeated = bool(info.get("bot_defeated", False))
        approach_reward = float(info["approach_reward"])
        attack_selected = bool(info["attack_selected"])
        valid_attack_attempt = bool(info["valid_attack_attempt"])
        invalid_attack = bool(info["invalid_attack"])
        tracking_failure = bool(info["tracking_failure"])
        movement_not_settled = bool(info["movement_not_settled"])
        cooldown_blocked = bool(info["cooldown_blocked"])
        confirmed_hit = damage > 0
        entered_attack_range = bool(info["entered_attack_range"])

        for key, value_to_append in {
            "observations": observation.copy(),
            "actions": action,
            "rewards": reward,
            "values": value,
            "log_probabilities": log_probability,
            "next_values": next_value,
            "terminated_flags": terminated,
            "episode_done_flags": episode_done,
            "damage_dealt": damage,
            "damage_taken": damage_taken,
            "bot_defeats": float(bot_defeated),
            "approach_rewards": approach_reward,
            "attack_selections": float(attack_selected),
            "valid_attack_attempts": float(valid_attack_attempt),
            "invalid_attacks": float(invalid_attack),
            "target_tracking_failures": float(tracking_failure),
            "movement_settle_failures": float(movement_not_settled),
            "cooldown_blocked_attacks": float(cooldown_blocked),
            "confirmed_hits": float(confirmed_hit),
        }.items():
            buffers[key].append(value_to_append)

        episode_state.episode_return += reward
        episode_state.episode_length += 1
        episode_state.damage_dealt += damage
        episode_state.damage_taken += damage_taken
        episode_state.bot_defeats += int(bot_defeated)
        episode_state.attack_selections += int(attack_selected)
        episode_state.valid_attack_attempts += int(valid_attack_attempt)
        episode_state.invalid_attacks += int(invalid_attack)
        episode_state.target_tracking_failures += int(tracking_failure)
        episode_state.movement_settle_failures += int(movement_not_settled)
        episode_state.cooldown_blocked_attacks += int(cooldown_blocked)
        episode_state.confirmed_hits += int(confirmed_hit)
        episode_state.range_entries += int(entered_attack_range)
        observation = next_observation

        if episode_done:
            _finish_episode(
                tf=tf,
                environment=environment,
                info=info,
                episode_state=episode_state,
                episode_logger=episode_logger,
                summary_writer=summary_writer,
            )
            observation, _ = environment.reset()
            episode_state.reset_episode()

    reward_array = np.asarray(buffers["rewards"], dtype=np.float32)
    value_array = np.asarray(buffers["values"], dtype=np.float32)
    next_value_array = np.asarray(buffers["next_values"], dtype=np.float32)
    terminated_array = np.asarray(
        buffers["terminated_flags"], dtype=np.float32
    )
    done_array = np.asarray(
        buffers["episode_done_flags"], dtype=np.float32
    )
    deltas = (
        reward_array
        + gamma * next_value_array * (1.0 - terminated_array)
        - value_array
    )
    advantages = np.zeros_like(reward_array)
    gae = 0.0
    for index in range(rollout_size - 1, -1, -1):
        gae = (
            deltas[index]
            + gamma * gae_lambda * (1.0 - done_array[index]) * gae
        )
        advantages[index] = gae
    returns = advantages + value_array
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    rollout = {
        "observations": np.asarray(buffers["observations"], dtype=np.float32),
        "actions": np.asarray(buffers["actions"], dtype=np.int32),
        "rewards": reward_array,
        "values": value_array,
        "log_probabilities": np.asarray(
            buffers["log_probabilities"], dtype=np.float32
        ),
        "advantages": advantages,
        "returns": returns,
        "damage_dealt": np.asarray(
            buffers["damage_dealt"], dtype=np.float32
        ),
        "damage_taken": np.asarray(
            buffers["damage_taken"], dtype=np.float32
        ),
        "bot_defeats": np.asarray(
            buffers["bot_defeats"], dtype=np.float32
        ),
        "approach_rewards": np.asarray(
            buffers["approach_rewards"], dtype=np.float32
        ),
        "attack_selections": np.asarray(
            buffers["attack_selections"], dtype=np.float32
        ),
        "valid_attack_attempts": np.asarray(
            buffers["valid_attack_attempts"], dtype=np.float32
        ),
        "invalid_attacks": np.asarray(
            buffers["invalid_attacks"], dtype=np.float32
        ),
        "target_tracking_failures": np.asarray(
            buffers["target_tracking_failures"], dtype=np.float32
        ),
        "movement_settle_failures": np.asarray(
            buffers["movement_settle_failures"], dtype=np.float32
        ),
        "cooldown_blocked_attacks": np.asarray(
            buffers["cooldown_blocked_attacks"], dtype=np.float32
        ),
        "confirmed_hits": np.asarray(
            buffers["confirmed_hits"], dtype=np.float32
        ),
    }
    return rollout, observation, episode_state


def _finish_episode(
    *,
    tf: Any,
    environment: StageTwoStationaryCombatEnv,
    info: dict[str, Any],
    episode_state: StageTwoEpisodeState,
    episode_logger: "StageTwoEpisodeCsvLogger",
    summary_writer: Any,
) -> None:
    episode_state.episode_number += 1
    success = bool(info["success"])
    episode_logger.write(
        episode=episode_state.episode_number,
        episode_return=episode_state.episode_return,
        length=episode_state.episode_length,
        success=success,
        final_distance=float(info["distance_to_target"]),
        target_health=float(info["target_health"]),
        damage_dealt=episode_state.damage_dealt,
        damage_taken=episode_state.damage_taken,
        bot_defeats=episode_state.bot_defeats,
        attack_selections=episode_state.attack_selections,
        valid_attack_attempts=episode_state.valid_attack_attempts,
        invalid_attacks=episode_state.invalid_attacks,
        target_tracking_failures=episode_state.target_tracking_failures,
        movement_settle_failures=episode_state.movement_settle_failures,
        cooldown_blocked_attacks=episode_state.cooldown_blocked_attacks,
        confirmed_hits=episode_state.confirmed_hits,
        range_entries=episode_state.range_entries,
    )
    with summary_writer.as_default():
        for name, value in {
            "return": episode_state.episode_return,
            "length": episode_state.episode_length,
            "success": float(success),
            "damage_dealt": episode_state.damage_dealt,
            "damage_taken": episode_state.damage_taken,
            "bot_defeats": episode_state.bot_defeats,
            "invalid_attacks": episode_state.invalid_attacks,
            "target_tracking_failures": episode_state.target_tracking_failures,
            "movement_settle_failures": episode_state.movement_settle_failures,
            "attack_selections": episode_state.attack_selections,
            "valid_attack_attempts": episode_state.valid_attack_attempts,
            "confirmed_hits": episode_state.confirmed_hits,
        }.items():
            tf.summary.scalar(
                f"episode/{name}",
                value,
                step=episode_state.episode_number,
            )


class StageTwoEpisodeCsvLogger:
    def __init__(self, path: Any) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "episode",
                "return",
                "length",
                "success",
                "final_distance",
                "target_health",
                "damage_dealt",
                "damage_taken",
                "bot_defeats",
                "attack_selections",
                "valid_attack_attempts",
                "invalid_attacks",
                "target_tracking_failures",
                "movement_settle_failures",
                "cooldown_blocked_attacks",
                "confirmed_hits",
                "range_entries",
            ]
        )

    def write(
        self,
        *,
        episode: int,
        episode_return: float,
        length: int,
        success: bool,
        final_distance: float,
        target_health: float,
        damage_dealt: float,
        damage_taken: float,
        bot_defeats: int,
        attack_selections: int,
        valid_attack_attempts: int,
        invalid_attacks: int,
        target_tracking_failures: int,
        movement_settle_failures: int,
        cooldown_blocked_attacks: int,
        confirmed_hits: int,
        range_entries: int,
    ) -> None:
        self._writer.writerow(
            [
                episode,
                episode_return,
                length,
                int(success),
                final_distance,
                target_health,
                damage_dealt,
                damage_taken,
                bot_defeats,
                attack_selections,
                valid_attack_attempts,
                invalid_attacks,
                target_tracking_failures,
                movement_settle_failures,
                cooldown_blocked_attacks,
                confirmed_hits,
                range_entries,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
