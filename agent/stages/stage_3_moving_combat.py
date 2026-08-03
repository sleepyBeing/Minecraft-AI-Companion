"""Stage three: defeat a moving zombie in a flat, hazard-free arena."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
    StageTwoStationaryCombatEnv,
)


class StageThreeMovingCombatEnv(StageTwoStationaryCombatEnv):

    BASE_BOT_POSITION = np.array([4.5, 7.5], dtype=np.float32)
    BASE_TARGET_POSITION = np.array([10.5, 7.5], dtype=np.float32)
    START_POSITION_JITTER = 0.35
    START_YAW_JITTER = 0.10

    ZOMBIE_MOVE_SPEED = 2.3
    ZOMBIE_ATTACK_RANGE = 1.8
    ZOMBIE_ATTACK_RANGE_EPSILON = 1e-4
    ZOMBIE_ATTACK_DAMAGE = 3.0
    ZOMBIE_ATTACK_COOLDOWN_SECONDS = 1.0

    DAMAGE_TAKEN_PENALTY_SCALE = 0.5

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.next_zombie_attack_time = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        supplied_options = dict(options or {})
        gym.Env.reset(self, seed=seed)

        has_bot_position = "bot_position" in supplied_options
        has_target_position = "target_position" in supplied_options
        if has_bot_position != has_target_position:
            raise ValueError(
                "bot_position and target_position must be provided together"
            )

        if not has_bot_position:
            supplied_options["bot_position"] = (
                self.BASE_BOT_POSITION
                + self.np_random.uniform(
                    -self.START_POSITION_JITTER,
                    self.START_POSITION_JITTER,
                    size=2,
                )
            ).tolist()
            supplied_options["target_position"] = (
                self.BASE_TARGET_POSITION
                + self.np_random.uniform(
                    -self.START_POSITION_JITTER,
                    self.START_POSITION_JITTER,
                    size=2,
                )
            ).tolist()

        if "bot_yaw" not in supplied_options:
            bot_position = np.asarray(
                supplied_options["bot_position"], dtype=np.float32
            )
            target_position = np.asarray(
                supplied_options["target_position"], dtype=np.float32
            )
            direction = target_position - bot_position
            facing_yaw = np.arctan2(-direction[0], -direction[1])
            supplied_options["bot_yaw"] = float(
                facing_yaw
                + self.np_random.uniform(
                    -self.START_YAW_JITTER,
                    self.START_YAW_JITTER,
                )
            )

        observation, info = super().reset(
            seed=seed,
            options=supplied_options,
        )
        self.next_zombie_attack_time = 0.0
        info.update(
            {
                "damage_taken": 0.0,
                "damage_taken_penalty": 0.0,
                "bot_defeated": False,
            }
        )
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        del observation

        damage_taken = 0.0
        if not terminated and not truncated:
            damage_taken = self._advance_zombie()

        damage_taken_penalty = (
            damage_taken * self.DAMAGE_TAKEN_PENALTY_SCALE
        )
        reward -= damage_taken_penalty

        bot_defeated = self.bot_defeated or self.bot_health <= 0
        terminated = terminated or bot_defeated
        truncated = truncated and not terminated
        success = not self.target_alive

        info.update(self._get_info())
        info.update(
            {
                "damage_taken": damage_taken,
                "damage_taken_penalty": damage_taken_penalty,
                "bot_defeated": bot_defeated,
                "success": success,
                "source": "simulation",
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _advance_zombie(self) -> float:
        offset = self.bot_position - self.target_position
        distance = float(np.linalg.norm(offset))
        if (
            distance
            > self.ZOMBIE_ATTACK_RANGE + self.ZOMBIE_ATTACK_RANGE_EPSILON
            and distance > 0
        ):
            movement = min(
                self.ZOMBIE_MOVE_SPEED * self.STEP_SECONDS,
                distance - self.ZOMBIE_ATTACK_RANGE,
            )
            self.target_position = np.clip(
                self.target_position + offset / distance * movement,
                0.5,
                self.ROOM_SIZE - 0.5,
            ).astype(np.float32)

        if (
            self._distance()
            <= self.ZOMBIE_ATTACK_RANGE + self.ZOMBIE_ATTACK_RANGE_EPSILON
            and self.elapsed_seconds >= self.next_zombie_attack_time
        ):
            damage = min(self.ZOMBIE_ATTACK_DAMAGE, self.bot_health)
            self.bot_health -= damage
            self.next_zombie_attack_time = (
                self.elapsed_seconds + self.ZOMBIE_ATTACK_COOLDOWN_SECONDS
            )
            return float(damage)
        return 0.0


class LiveStageThreeMovingCombatEnv(
    LiveStageTwoStationaryCombatEnv,
    StageThreeMovingCombatEnv,
):
    """Moderate penalty for taking damage, not too big to encourage trading damage"""

    BRIDGE_STAGE = "stage3"
    BRIDGE_STAGE_NAME = "stage-three"

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        health_before_action = self.bot_health
        observation, reward, terminated, truncated, info = super().step(action)

        damage_taken = max(0.0, health_before_action - self.bot_health)
        damage_taken_penalty = (
            damage_taken * self.DAMAGE_TAKEN_PENALTY_SCALE
        )
        reward -= damage_taken_penalty
        bot_defeated = self.bot_defeated or self.bot_health <= 0
        terminated = terminated or bot_defeated
        truncated = truncated and not terminated
        success = not self.target_alive
        info.update(
            {
                "damage_taken": damage_taken,
                "damage_taken_penalty": damage_taken_penalty,
                "bot_defeated": bot_defeated,
                "success": success,
                "source": "minecraft",
            }
        )
        return observation, float(reward), terminated, truncated, info
