"""Rollout collection and episode logging for Stage Four."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from agent.stages.stage_2_stationary_combat import StationaryCombatAction
from agent.stages.stage_4_retreat import StageFourRetreatEnv


class StageFourEpisodeState:
    def __init__(self, starting_health: float) -> None:
        self.episode_number = 0
        self.reset_episode(starting_health)

    def reset_episode(self, starting_health: float) -> None:
        self.starting_health = starting_health
        self.survival_mode = (
            starting_health
            in StageFourRetreatEnv.SURVIVAL_STARTING_HEALTH_OPTIONS
        )
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
        self.retreat_actions = 0
        self.safe_area_actions = 0
        self.wait_actions = 0
        self.low_health_steps = 0
        self.cover_steps = 0
        self.cover_entries = 0
        self.retreat_reward = 0.0
        self.cover_progress_reward = 0.0
        self.cover_reward = 0.0
        self.cover_maintenance_reward = 0.0
        self.survival_step_reward = 0.0
        self.survival_attack_penalty = 0.0
        self.survival_reward = 0.0
        self.death_penalty = 0.0


def collect_stage_four_rollout(
    *,
    tf: Any,
    actor: Any,
    critic: Any,
    environment: StageFourRetreatEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_state: StageFourEpisodeState,
    episode_logger: "StageFourEpisodeCsvLogger",
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageFourEpisodeState]:
    names = (
        "observations",
        "actions",
        "rewards",
        "values",
        "log_probabilities",
        "next_values",
        "terminated_flags",
        "episode_done_flags",
        "damage_dealt",
        "damage_taken",
        "bot_defeats",
        "approach_rewards",
        "attack_selections",
        "valid_attack_attempts",
        "invalid_attacks",
        "target_tracking_failures",
        "movement_settle_failures",
        "cooldown_blocked_attacks",
        "confirmed_hits",
        "retreat_actions",
        "safe_area_actions",
        "wait_actions",
        "low_health_steps",
        "survival_mode_steps",
        "cover_occluded_steps",
        "cover_streaks",
        "cover_steps",
        "cover_entries",
        "retreat_rewards",
        "cover_progress_rewards",
        "cover_rewards",
        "cover_maintenance_rewards",
        "cover_abandonment_penalties",
        "survival_step_rewards",
        "survival_attack_penalties",
        "survival_rewards",
        "death_penalties",
    )
    buffers: dict[str, list[Any]] = {name: [] for name in names}
    observation = initial_observation

    for _ in range(rollout_size):
        observation_tensor = tf.convert_to_tensor(
            observation[None, :], dtype=tf.float32
        )
        logits = actor(observation_tensor, training=False)
        value = float(critic(observation_tensor, training=False)[0, 0].numpy())
        action = int(tf.random.categorical(logits, 1)[0, 0].numpy())
        log_probability = float(tf.nn.log_softmax(logits)[0, action].numpy())

        next_observation, reward, terminated, truncated, info = environment.step(action)
        next_value = float(
            critic(
                tf.convert_to_tensor(next_observation[None, :], dtype=tf.float32),
                training=False,
            )[0, 0].numpy()
        )
        episode_done = terminated or truncated
        damage_dealt = float(info["damage_dealt"])
        damage_taken = float(info["damage_taken"])
        bot_defeated = bool(info["bot_defeated"])
        retreat_action = action == int(StationaryCombatAction.RETREAT)
        safe_area_action = action == int(StationaryCombatAction.MOVE_TO_SAFE_AREA)
        wait_action = action == int(StationaryCombatAction.WAIT)
        low_health = bool(info["low_health"])
        survival_mode = bool(info["survival_mode"])
        cover_occluded = bool(info["cover_occluded"])
        in_cover = bool(info["in_cover"])
        cover_reward = float(info["cover_reward"])

        values_to_append = {
            "observations": observation.copy(),
            "actions": action,
            "rewards": reward,
            "values": value,
            "log_probabilities": log_probability,
            "next_values": next_value,
            "terminated_flags": terminated,
            "episode_done_flags": episode_done,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "bot_defeats": float(bot_defeated),
            "approach_rewards": float(info["approach_reward"]),
            "attack_selections": float(info["attack_selected"]),
            "valid_attack_attempts": float(info["valid_attack_attempt"]),
            "invalid_attacks": float(info["invalid_attack"]),
            "target_tracking_failures": float(info["tracking_failure"]),
            "movement_settle_failures": float(info["movement_not_settled"]),
            "cooldown_blocked_attacks": float(info["cooldown_blocked"]),
            "confirmed_hits": float(damage_dealt > 0),
            "retreat_actions": float(retreat_action),
            "safe_area_actions": float(safe_area_action),
            "wait_actions": float(wait_action),
            "low_health_steps": float(low_health),
            "survival_mode_steps": float(survival_mode),
            "cover_occluded_steps": float(cover_occluded),
            "cover_streaks": float(info["cover_streak"]),
            "cover_steps": float(in_cover),
            "cover_entries": float(cover_reward > 0),
            "retreat_rewards": float(info["retreat_progress_reward"]),
            "cover_progress_rewards": float(info["cover_progress_reward"]),
            "cover_rewards": cover_reward,
            "cover_maintenance_rewards": float(
                info["cover_maintenance_reward"]
            ),
            "cover_abandonment_penalties": float(
                info["cover_abandonment_penalty"]
            ),
            "survival_step_rewards": float(info["survival_step_reward"]),
            "survival_attack_penalties": float(
                info["survival_attack_penalty"]
            ),
            "survival_rewards": float(info["survival_reward"]),
            "death_penalties": float(info["death_penalty"]),
        }
        for name, item in values_to_append.items():
            buffers[name].append(item)

        episode_state.episode_return += reward
        episode_state.episode_length += 1
        episode_state.damage_dealt += damage_dealt
        episode_state.damage_taken += damage_taken
        episode_state.bot_defeats += int(bot_defeated)
        episode_state.attack_selections += int(info["attack_selected"])
        episode_state.valid_attack_attempts += int(info["valid_attack_attempt"])
        episode_state.invalid_attacks += int(info["invalid_attack"])
        episode_state.target_tracking_failures += int(info["tracking_failure"])
        episode_state.movement_settle_failures += int(info["movement_not_settled"])
        episode_state.cooldown_blocked_attacks += int(info["cooldown_blocked"])
        episode_state.confirmed_hits += int(damage_dealt > 0)
        episode_state.range_entries += int(info["entered_attack_range"])
        episode_state.retreat_actions += int(retreat_action)
        episode_state.safe_area_actions += int(safe_area_action)
        episode_state.wait_actions += int(wait_action)
        episode_state.low_health_steps += int(low_health)
        episode_state.cover_steps += int(in_cover)
        episode_state.cover_entries += int(cover_reward > 0)
        episode_state.retreat_reward += float(info["retreat_progress_reward"])
        episode_state.cover_progress_reward += float(
            info["cover_progress_reward"]
        )
        episode_state.cover_reward += cover_reward
        episode_state.cover_maintenance_reward += float(
            info["cover_maintenance_reward"]
        )
        episode_state.survival_step_reward += float(
            info["survival_step_reward"]
        )
        episode_state.survival_attack_penalty += float(
            info["survival_attack_penalty"]
        )
        episode_state.survival_reward += float(info["survival_reward"])
        episode_state.death_penalty += float(info["death_penalty"])
        observation = next_observation

        if episode_done:
            _finish_episode(
                tf=tf,
                info=info,
                state=episode_state,
                logger=episode_logger,
                summary_writer=summary_writer,
            )
            observation, reset_info = environment.reset()
            episode_state.reset_episode(float(reset_info["starting_bot_health"]))

    reward_array = np.asarray(buffers["rewards"], dtype=np.float32)
    value_array = np.asarray(buffers["values"], dtype=np.float32)
    next_value_array = np.asarray(buffers["next_values"], dtype=np.float32)
    terminated_array = np.asarray(buffers["terminated_flags"], dtype=np.float32)
    done_array = np.asarray(buffers["episode_done_flags"], dtype=np.float32)
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
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rollout: dict[str, np.ndarray] = {
        "observations": np.asarray(buffers["observations"], dtype=np.float32),
        "actions": np.asarray(buffers["actions"], dtype=np.int32),
        "rewards": reward_array,
        "values": value_array,
        "log_probabilities": np.asarray(
            buffers["log_probabilities"], dtype=np.float32
        ),
        "advantages": advantages,
        "returns": returns,
    }
    for name in names[8:]:
        rollout[name] = np.asarray(buffers[name], dtype=np.float32)
    return rollout, observation, episode_state


def _finish_episode(
    *,
    tf: Any,
    info: dict[str, Any],
    state: StageFourEpisodeState,
    logger: "StageFourEpisodeCsvLogger",
    summary_writer: Any,
) -> None:
    state.episode_number += 1
    killed_target = not bool(info["target_alive"])
    survived_timeout = float(info["survival_reward"]) > 0
    if state.bot_defeats:
        outcome = "death"
    elif killed_target:
        outcome = "kill"
    elif survived_timeout:
        outcome = "survived_timeout"
    elif state.target_tracking_failures:
        outcome = "tracking_failure"
    else:
        outcome = "timeout"

    logger.write(
        episode=state.episode_number,
        episode_return=state.episode_return,
        length=state.episode_length,
        success=bool(info["success"]),
        outcome=outcome,
        starting_health=state.starting_health,
        final_health=float(info["bot_health"]),
        final_distance=float(info["distance_to_target"]),
        target_health=float(info["target_health"]),
        state=state,
    )
    with summary_writer.as_default():
        for name, value in {
            "return": state.episode_return,
            "length": state.episode_length,
            "success": float(info["success"]),
            "kill": float(killed_target),
            "survived_timeout": float(survived_timeout),
            "starting_health": state.starting_health,
            "survival_mode": float(state.survival_mode),
            "damage_dealt": state.damage_dealt,
            "damage_taken": state.damage_taken,
            "bot_defeats": state.bot_defeats,
            "invalid_attacks": state.invalid_attacks,
            "retreat_actions": state.retreat_actions,
            "safe_area_actions": state.safe_area_actions,
            "cover_steps": state.cover_steps,
            "cover_entries": state.cover_entries,
            "retreat_reward": state.retreat_reward,
            "cover_progress_reward": state.cover_progress_reward,
            "cover_reward": state.cover_reward,
            "cover_maintenance_reward": state.cover_maintenance_reward,
            "survival_step_reward": state.survival_step_reward,
            "survival_attack_penalty": state.survival_attack_penalty,
            "survival_reward": state.survival_reward,
            "death_penalty": state.death_penalty,
        }.items():
            tf.summary.scalar(f"episode/{name}", value, step=state.episode_number)


class StageFourEpisodeCsvLogger:
    COLUMNS = (
        "episode",
        "return",
        "length",
        "success",
        "outcome",
        "starting_health",
        "scenario",
        "final_health",
        "final_distance",
        "target_health",
        "damage_dealt",
        "damage_taken",
        "bot_defeats",
        "attack_selections",
        "valid_attack_attempts",
        "invalid_attacks",
        "confirmed_hits",
        "retreat_actions",
        "safe_area_actions",
        "wait_actions",
        "low_health_steps",
        "cover_steps",
        "cover_entries",
        "retreat_reward",
        "cover_progress_reward",
        "cover_reward",
        "cover_maintenance_reward",
        "survival_step_reward",
        "survival_attack_penalty",
        "survival_reward",
        "death_penalty",
        "target_tracking_failures",
        "movement_settle_failures",
        "cooldown_blocked_attacks",
        "range_entries",
    )

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def write(
        self,
        *,
        episode: int,
        episode_return: float,
        length: int,
        success: bool,
        outcome: str,
        starting_health: float,
        final_health: float,
        final_distance: float,
        target_health: float,
        state: StageFourEpisodeState,
    ) -> None:
        self._writer.writerow(
            [
                episode,
                episode_return,
                length,
                int(success),
                outcome,
                starting_health,
                "survival" if state.survival_mode else "combat_retreat",
                final_health,
                final_distance,
                target_health,
                state.damage_dealt,
                state.damage_taken,
                state.bot_defeats,
                state.attack_selections,
                state.valid_attack_attempts,
                state.invalid_attacks,
                state.confirmed_hits,
                state.retreat_actions,
                state.safe_area_actions,
                state.wait_actions,
                state.low_health_steps,
                state.cover_steps,
                state.cover_entries,
                state.retreat_reward,
                state.cover_progress_reward,
                state.cover_reward,
                state.cover_maintenance_reward,
                state.survival_step_reward,
                state.survival_attack_penalty,
                state.survival_reward,
                state.death_penalty,
                state.target_tracking_failures,
                state.movement_settle_failures,
                state.cooldown_blocked_attacks,
                state.range_entries,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
