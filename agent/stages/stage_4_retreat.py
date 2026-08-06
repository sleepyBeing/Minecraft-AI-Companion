"""Stage four: fight when healthy and retreat behind cover when health is low."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
    StationaryCombatAction,
)
from agent.stages.stage_3_moving_combat import StageThreeMovingCombatEnv


class StageFourRetreatEnv(StageThreeMovingCombatEnv):

    STARTING_HEALTH_OPTIONS = (20.0, 10.0, 5.0)
    LOW_HEALTH_THRESHOLD = 10.0
    COVER_RECTANGLES = (
        (7.0, 8.0, 2.0, 5.0),
        (7.0, 8.0, 10.0, 13.0),
    )
    SAFE_OFFSET = 1.75

    RETREAT_PROGRESS_REWARD_SCALE = 1.5
    COVER_DISCOVERY_REWARD = 3.0
    HEALTHY_COVER_REWARD = 0.5
    HEALTHY_RETREAT_ACTION_PENALTY = 0.05
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
        self.cover_bonus_awarded = False

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
        self.bot_health = requested_health
        self.bot_defeated = False
        self.cover_bonus_awarded = False
        info.update(self._get_info())
        info.update(
            {
                "starting_bot_health": self.starting_bot_health,
                "retreat_progress_reward": 0.0,
                "cover_reward": 0.0,
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
        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        return self._apply_stage_four_rewards(
            action=action,
            health_before_action=health_before_action,
            in_cover_before=in_cover_before,
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
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        source: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        low_health = health_before_action <= self.LOW_HEALTH_THRESHOLD
        retreat_progress_reward = 0.0

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

        in_cover_after = self._is_in_cover()
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

        success = not self.target_alive or timed_out_alive
        info.update(self._get_info())
        info.update(
            {
                "starting_bot_health": self.starting_bot_health,
                "low_health": low_health,
                "retreat_progress_reward": retreat_progress_reward,
                "cover_reward": cover_reward,
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

        if self._point_inside_cover(self.bot_position):
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
            if self._point_inside_cover(candidate):
                alternatives = (
                    self.target_position + np.array([movement[0], 0.0]),
                    self.target_position + np.array([0.0, movement[1]]),
                )
                valid = [
                    point
                    for point in alternatives
                    if not self._point_inside_cover(point)
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
            not self._is_in_cover()
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
                float(self._is_in_cover()),
                float(self.bot_health <= self.LOW_HEALTH_THRESHOLD),
            ],
            dtype=np.float32,
        )
        return np.concatenate((stage_three, stage_four)).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        safe_position = self._nearest_safe_position()
        info.update(
            {
                "safe_position": safe_position.copy(),
                "distance_to_safe_position": float(
                    np.linalg.norm(safe_position - self.bot_position)
                ),
                "starting_bot_health": self.starting_bot_health,
                "in_cover": self._is_in_cover(),
                "low_health": self.bot_health <= self.LOW_HEALTH_THRESHOLD,
            }
        )
        return info

    def render(self) -> str:
        grid = np.full((15, 15), ".", dtype="<U1")
        for min_x, max_x, min_z, max_z in self.COVER_RECTANGLES:
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
        candidates = [
            self._safe_point_behind(rectangle)
            for rectangle in self.COVER_RECTANGLES
        ]
        return min(
            candidates,
            key=lambda point: float(np.linalg.norm(point - self.bot_position)),
        ).copy()

    def _safe_point_behind(
        self, rectangle: tuple[float, float, float, float]
    ) -> np.ndarray:
        min_x, max_x, min_z, max_z = rectangle
        center = np.array(
            [(min_x + max_x) / 2.0, (min_z + max_z) / 2.0],
            dtype=np.float32,
        )
        direction = center - self.target_position
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction /= length
        return np.clip(
            center + direction * self.SAFE_OFFSET,
            0.5,
            self.ROOM_SIZE - 0.5,
        ).astype(np.float32)

    def _is_in_cover(self) -> bool:
        return any(
            self._segment_intersects_rectangle(
                self.bot_position,
                self.target_position,
                rectangle,
            )
            for rectangle in self.COVER_RECTANGLES
        )

    def _point_inside_cover(self, point: np.ndarray) -> bool:
        return any(
            min_x <= point[0] <= max_x and min_z <= point[1] <= max_z
            for min_x, max_x, min_z, max_z in self.COVER_RECTANGLES
        )

    @staticmethod
    def _segment_intersects_rectangle(
        start: np.ndarray,
        end: np.ndarray,
        rectangle: tuple[float, float, float, float],
    ) -> bool:
        min_x, max_x, min_z, max_z = rectangle
        delta = end - start
        minimum = 0.0
        maximum = 1.0
        for origin, change, low, high in (
            (start[0], delta[0], min_x, max_x),
            (start[1], delta[1], min_z, max_z),
        ):
            if abs(float(change)) < 1e-9:
                if origin < low or origin > high:
                    return False
                continue
            first = (low - origin) / change
            second = (high - origin) / change
            minimum = max(minimum, float(min(first, second)))
            maximum = min(maximum, float(max(first, second)))
            if minimum > maximum:
                return False
        return True


class LiveStageFourRetreatEnv(
    LiveStageTwoStationaryCombatEnv,
    StageFourRetreatEnv,
):

    BRIDGE_STAGE = "stage4"
    BRIDGE_STAGE_NAME = "stage-four"

    def __init__(
        self,
        bridge_url: str = "ws://127.0.0.1:8765",
        *,
        bridge_timeout: float = 15.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            bridge_url=bridge_url,
            bridge_timeout=bridge_timeout,
            render_mode=render_mode,
        )
        self.server_safe_position = np.array([7.5, 3.5], dtype=np.float32)
        self.server_in_cover = False

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        health_before_action = self.bot_health
        in_cover_before = self._is_in_cover()
        observation, reward, terminated, truncated, info = (
            LiveStageTwoStationaryCombatEnv.step(self, action)
        )
        del observation

        damage_taken = max(0.0, health_before_action - self.bot_health)
        damage_taken_penalty = (
            damage_taken * self.DAMAGE_TAKEN_PENALTY_SCALE
        )
        reward -= damage_taken_penalty
        bot_defeated = self.bot_defeated or self.bot_health <= 0
        terminated = terminated or bot_defeated
        truncated = truncated and not terminated
        info.update(
            {
                "damage_taken": damage_taken,
                "damage_taken_penalty": damage_taken_penalty,
                "bot_defeated": bot_defeated,
            }
        )
        return self._apply_stage_four_rewards(
            action=action,
            health_before_action=health_before_action,
            in_cover_before=in_cover_before,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            source="minecraft",
        )

    def _apply_live_state(self, state: dict[str, Any]) -> None:
        super()._apply_live_state(state)
        try:
            safe_position = np.asarray(state["safePosition"], dtype=np.float32)
            if safe_position.shape != (2,):
                raise ValueError("safePosition must contain [x, z]")
            self.server_safe_position = safe_position
            self.server_in_cover = bool(state["inCover"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid stage-four cover state from Mineflayer: {state!r}"
            ) from error

    def _nearest_safe_position(self) -> np.ndarray:
        return self.server_safe_position.copy()

    def _is_in_cover(self) -> bool:
        return self.server_in_cover
