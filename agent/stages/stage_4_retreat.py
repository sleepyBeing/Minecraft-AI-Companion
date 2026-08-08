"""Stage four: fight when healthy and retreat behind cover when health is low."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from agent.stages.stage_4_cover import (
    COVER_RECTANGLES,
    choose_cover_plan,
    is_geometrically_occluded,
    point_inside_cover,
)
from agent.stages.stage_2_stationary_combat import (
    StationaryCombatAction,
)
from agent.stages.stage_3_moving_combat import StageThreeMovingCombatEnv


class StageFourRetreatEnv(StageThreeMovingCombatEnv):

    STARTING_HEALTH_OPTIONS = (20.0, 10.0, 5.0)
    SURVIVAL_STARTING_HEALTH_OPTIONS = (10.0, 5.0)
    LOW_HEALTH_THRESHOLD = 10.0
    COVER_CONFIRMATION_STEPS = 3

    RETREAT_PROGRESS_REWARD_SCALE = 1.5
    COVER_PROGRESS_REWARD_SCALE = 1.0
    COVER_DISCOVERY_REWARD = 3.0
    COVER_MAINTENANCE_REWARD = 0.02
    HEALTHY_COVER_REWARD = 0.5
    HEALTHY_RETREAT_ACTION_PENALTY = 0.05
    SURVIVAL_STEP_REWARD = 0.02
    SURVIVAL_ATTACK_PENALTY = 0.05
    SURVIVAL_REWARD = 20.0
    DEATH_PENALTY = 25.0

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        stage_three_low = self.observation_space.low.copy()
        stage_three_high = self.observation_space.high.copy()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    stage_three_low,
                    np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                )
            ),
            high=np.concatenate(
                (
                    stage_three_high,
                    np.ones(5, dtype=np.float32),
                )
            ),
            dtype=np.float32,
        )
        self.starting_bot_health = self.MAX_HEALTH
        self.survival_mode = False
        self.cover_bonus_awarded = False
        self.cover_streak = 0
        self.confirmed_in_cover = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        supplied_options = dict(options or {})
        requested_health = supplied_options.get("bot_health")
        observation, info = super().reset(seed=seed, options=supplied_options)
        del observation

        if requested_health is None:
            requested_health = float(self.np_random.choice(self.STARTING_HEALTH_OPTIONS))
        requested_health = float(requested_health)
        if requested_health not in self.STARTING_HEALTH_OPTIONS:
            raise ValueError("bot_health must be one of 20, 10, or 5")

        self.starting_bot_health = requested_health
        self.survival_mode = (
            requested_health in self.SURVIVAL_STARTING_HEALTH_OPTIONS
        )
        self.bot_health = requested_health
        self.bot_defeated = False
        self.cover_bonus_awarded = False
        self.cover_streak = 0
        self.confirmed_in_cover = False
        info.update(self._get_info())
        info.update(
            {
                "starting_bot_health": self.starting_bot_health,
                "survival_mode": self.survival_mode,
                "scenario": self._scenario_name(),
                "retreat_progress_reward": 0.0,
                "cover_progress_reward": 0.0,
                "cover_distance_change": 0.0,
                "cover_reward": 0.0,
                "cover_maintenance_reward": 0.0,
                "survival_step_reward": 0.0,
                "survival_attack_penalty": 0.0,
                "survival_reward": 0.0,
                "death_penalty": 0.0,
                "success": False,
            }
        )
        return self._get_observation(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        health_before_action = self.bot_health
        in_cover_before = self._is_in_cover()
        safe_position_before = self._nearest_safe_position()
        safe_distance_before = float(
            np.linalg.norm(safe_position_before - self.bot_position)
        )
        if self.survival_mode:
            self.target_health = self.ZOMBIE_MAX_HEALTH
            self.target_alive = True
        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        if self.survival_mode:
            self.target_health = self.ZOMBIE_MAX_HEALTH
            self.target_alive = True
        self._update_cover_confirmation()
        return self._apply_stage_four_rewards(
            action=action,
            health_before_action=health_before_action,
            in_cover_before=in_cover_before,
            safe_position_before=safe_position_before,
            safe_distance_before=safe_distance_before,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            source="simulation",
        )

    def _apply_stage_four_rewards(
        self,
        *,
        action: int,
        health_before_action: float,
        in_cover_before: bool,
        safe_position_before: np.ndarray,
        safe_distance_before: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        source: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        low_health = health_before_action <= self.LOW_HEALTH_THRESHOLD
        retreat_progress_reward = 0.0
        survival_attack_penalty = 0.0

        if self.survival_mode:
            # Survival episodes cannot obtain positive return from the inherited Stage Three combat objective
            reward -= float(info["damage_dealt"]) * self.DAMAGE_REWARD_SCALE
            if bool(info["entered_attack_range"]):
                reward -= self.ATTACK_RANGE_ENTRY_REWARD
            if bool(info["attack_selected"]):
                survival_attack_penalty = self.SURVIVAL_ATTACK_PENALTY
                reward -= survival_attack_penalty
            info.update(
                {
                    "damage_dealt": 0.0,
                    "attack_landed": False,
                    "confirmed_kill": False,
                }
            )

        if low_health and self.target_alive:
            reward -= float(info["approach_reward"])
            retreat_progress_reward = (
                -float(info["distance_change"])
                * self.RETREAT_PROGRESS_REWARD_SCALE
            )
            reward += retreat_progress_reward
        elif action in (
            int(StationaryCombatAction.RETREAT),
            int(StationaryCombatAction.MOVE_TO_SAFE_AREA),
        ):
            reward -= self.HEALTHY_RETREAT_ACTION_PENALTY

        in_cover_after = self.confirmed_in_cover
        safe_distance_after = float(
            np.linalg.norm(safe_position_before - self.bot_position)
        )
        cover_distance_change = safe_distance_before - safe_distance_after
        cover_progress_reward = 0.0
        if low_health and self.target_alive and not in_cover_before:
            if (
                action == int(StationaryCombatAction.MOVE_TO_SAFE_AREA)
                and cover_distance_change > 0.0
                and retreat_progress_reward < 0.0
            ):
                # Moving laterally toward cover can temporarily let the
                # pursuing zombie close the gap. Do not punish a successful
                # safe-area step for that expected geometric tradeoff.
                reward -= retreat_progress_reward
                retreat_progress_reward = 0.0
            cover_progress_reward = (
                cover_distance_change * self.COVER_PROGRESS_REWARD_SCALE
            )
            reward += cover_progress_reward

        cover_reward = 0.0
        if (
            not self.cover_bonus_awarded
            and not in_cover_before
            and in_cover_after
        ):
            cover_reward = (
                self.COVER_DISCOVERY_REWARD
                if low_health
                else self.HEALTHY_COVER_REWARD
            )
            reward += cover_reward
            self.cover_bonus_awarded = True

        bot_defeated = bool(info.get("bot_defeated", False))
        cover_maintenance_reward = 0.0
        if (
            low_health
            and in_cover_after
            and self.target_alive
            and not bot_defeated
        ):
            cover_maintenance_reward = self.COVER_MAINTENANCE_REWARD
            reward += cover_maintenance_reward

        survival_step_reward = 0.0
        if self.survival_mode and self.target_alive and not bot_defeated:
            survival_step_reward = self.SURVIVAL_STEP_REWARD
            reward += survival_step_reward

        death_penalty = self.DEATH_PENALTY if bot_defeated else 0.0
        reward -= death_penalty

        tracking_failure = bool(info.get("tracking_failure", False))
        timed_out_alive = (
            truncated
            and not tracking_failure
            and not bot_defeated
            and self.target_alive
            and self.elapsed_seconds >= self.EPISODE_SECONDS
        )
        survival_reward = self.SURVIVAL_REWARD if timed_out_alive else 0.0
        reward += survival_reward

        success = (
            timed_out_alive
            if self.survival_mode
            else not self.target_alive or timed_out_alive
        )
        info.update(self._get_info())
        info.update(
            {
                "starting_bot_health": self.starting_bot_health,
                "survival_mode": self.survival_mode,
                "scenario": self._scenario_name(),
                "low_health": low_health,
                "retreat_progress_reward": retreat_progress_reward,
                "cover_progress_reward": cover_progress_reward,
                "cover_distance_change": cover_distance_change,
                "cover_reward": cover_reward,
                "cover_maintenance_reward": cover_maintenance_reward,
                "survival_step_reward": survival_step_reward,
                "survival_attack_penalty": survival_attack_penalty,
                "survival_reward": survival_reward,
                "death_penalty": death_penalty,
                "success": success,
                "source": source,
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _apply_movement_action(self, action: StationaryCombatAction) -> None:
        previous_position = self.bot_position.copy()
        if action == StationaryCombatAction.RETREAT:
            direction = self.bot_position - self.target_position
            self._move_in_world_direction(direction)
        elif action == StationaryCombatAction.MOVE_TO_SAFE_AREA:
            direction = self._nearest_safe_position() - self.bot_position
            self._move_in_world_direction(direction)
        else:
            super()._apply_movement_action(action)

        if point_inside_cover(self.bot_position):
            self.bot_position = previous_position

    def _move_in_world_direction(self, direction: np.ndarray) -> None:
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return
        unit = direction / length
        self.bot_position = np.clip(
            self.bot_position + unit * self.MOVE_SPEED * self.STEP_SECONDS,
            0.5,
            self.ROOM_SIZE - 0.5,
        ).astype(np.float32)
        self.bot_yaw = self._wrap_angle(float(np.arctan2(-unit[0], -unit[1])))

    def _advance_zombie(self) -> float:
        offset = self.bot_position - self.target_position
        distance = float(np.linalg.norm(offset))
        if (
            distance
            > self.ZOMBIE_ATTACK_RANGE + self.ZOMBIE_ATTACK_RANGE_EPSILON
            and distance > 0
        ):
            movement_length = min(
                self.ZOMBIE_MOVE_SPEED * self.STEP_SECONDS,
                distance - self.ZOMBIE_ATTACK_RANGE,
            )
            movement = offset / distance * movement_length
            candidate = self.target_position + movement
            if point_inside_cover(candidate):
                alternatives = (
                    self.target_position + np.array([movement[0], 0.0]),
                    self.target_position + np.array([0.0, movement[1]]),
                )
                valid = [
                    point
                    for point in alternatives
                    if not point_inside_cover(point)
                ]
                if valid:
                    candidate = min(
                        valid,
                        key=lambda point: float(
                            np.linalg.norm(self.bot_position - point)
                        ),
                    )
                else:
                    candidate = self.target_position
            self.target_position = np.clip(
                candidate, 0.5, self.ROOM_SIZE - 0.5
            ).astype(np.float32)

        if (
            not self._raw_cover_occlusion()
            and self._distance()
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

    def _get_observation(self) -> np.ndarray:
        stage_three = super()._get_observation()
        safe_position = self._nearest_safe_position()
        relative_safe = (safe_position - self.bot_position) / self.ROOM_SIZE
        safe_distance = float(np.linalg.norm(safe_position - self.bot_position))
        stage_four = np.array(
            [
                relative_safe[0],
                relative_safe[1],
                safe_distance / (np.sqrt(2.0) * self.ROOM_SIZE),
                float(self.confirmed_in_cover),
                float(self.survival_mode),
            ],
            dtype=np.float32,
        )
        return np.concatenate((stage_three, stage_four)).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        plan = self._cover_plan()
        safe_position = plan.navigation_position
        info.update(
            {
                "safe_position": safe_position.copy(),
                "protected_position": plan.protected_position.copy(),
                "active_cover": plan.cover_index,
                "distance_to_safe_position": float(
                    np.linalg.norm(safe_position - self.bot_position)
                ),
                "starting_bot_health": self.starting_bot_health,
                "survival_mode": self.survival_mode,
                "scenario": self._scenario_name(),
                "in_cover": self.confirmed_in_cover,
                "cover_occluded": self._raw_cover_occlusion(),
                "cover_streak": self.cover_streak,
                "low_health": self.bot_health <= self.LOW_HEALTH_THRESHOLD,
            }
        )
        return info

    def _scenario_name(self) -> str:
        return "survival" if self.survival_mode else "combat_retreat"

    def render(self) -> str:
        grid = np.full((15, 15), ".", dtype="<U1")
        for min_x, max_x, min_z, max_z in COVER_RECTANGLES:
            grid[
                int(min_z) : int(max_z),
                int(min_x) : int(max_x),
            ] = "#"
        if self.target_alive:
            target_x, target_z = np.floor(self.target_position).astype(int)
            grid[target_z, target_x] = "Z"
        bot_x, bot_z = np.floor(self.bot_position).astype(int)
        grid[bot_z, bot_x] = "B"
        return "\n".join("".join(row) for row in grid)

    def _nearest_safe_position(self) -> np.ndarray:
        return self._cover_plan().navigation_position.copy()

    def _protected_safe_position(self) -> np.ndarray:
        return self._cover_plan().protected_position.copy()

    def _cover_plan(self):
        return choose_cover_plan(
            self.bot_position,
            self.target_position,
            room_size=self.ROOM_SIZE,
        )

    def _raw_cover_occlusion(self) -> bool:
        return is_geometrically_occluded(
            self.bot_position,
            self.target_position,
        )

    def _is_in_cover(self) -> bool:
        return self.confirmed_in_cover

    def _update_cover_confirmation(self) -> None:
        if self._raw_cover_occlusion():
            self.cover_streak += 1
        else:
            self.cover_streak = 0
        self.confirmed_in_cover = (
            self.cover_streak >= self.COVER_CONFIRMATION_STEPS
        )
