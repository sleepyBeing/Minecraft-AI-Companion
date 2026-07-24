"""Stage-one Gymnasium environment for basic combat positioning.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PositioningAction(IntEnum):

    WAIT = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6


class BasicPositioningEnv(gym.Env[np.ndarray, int]):
    """A 15x15 room containing the bot and one stationary target.

    The episode succeeds when the bot gets within melee attack range. The
    60-second limit represents simulated game time, so training can run faster
    than real time.

    Observation (all ``float32``):
        0. bot x position, normalized to [-1, 1]
        1. bot z position, normalized to [-1, 1]
        2. sin(bot yaw)
        3. cos(bot yaw)
        4. bot health / 20 (always 1.0 in stage one)
        5. has equipment (always 0.0 in stage one)
        6. target x relative to bot, normalized by room size
        7. target z relative to bot, normalized by room size
        8. bot-to-target distance, normalized by the room diagonal
        9. whether the target is currently in attack range
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    ROOM_SIZE = 15.0
    MAX_HEALTH = 20.0
    ATTACK_RANGE = 3.0
    EPISODE_SECONDS = 60.0
    STEP_SECONDS = 0.1
    MOVE_SPEED = 4.3
    TURN_SPEED_RADIANS = np.pi

    PROGRESS_REWARD_SCALE = 1.0
    ATTACK_RANGE_REWARD = 0.5
    TIME_PENALTY = 0.01

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(PositioningAction))
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.ones(10, dtype=np.float32),
            dtype=np.float32,
        )

        self.bot_position = np.zeros(2, dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        self.bot_yaw = 0.0
        self.bot_health = self.MAX_HEALTH
        self.has_equipment = False
        self.elapsed_seconds = 0.0
        self.previous_distance = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        self.bot_position = self._position_from_options(options, "bot_position")
        self.target_position = self._position_from_options(options, "target_position")

        if "bot_position" not in options and "target_position" not in options:
            self._randomize_separated_positions()
        elif self._distance() <= self.ATTACK_RANGE:
            raise ValueError("Initial bot and target positions must be outside attack range")

        self.bot_yaw = float(
            options.get("bot_yaw", self.np_random.uniform(-np.pi, np.pi))
        )
        self.bot_yaw = self._wrap_angle(self.bot_yaw)
        self.bot_health = self.MAX_HEALTH
        self.has_equipment = False
        self.elapsed_seconds = 0.0
        self.previous_distance = self._distance()

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 6")

        old_position = self.bot_position.copy()
        self._apply_action(PositioningAction(action))
        self.elapsed_seconds += self.STEP_SECONDS

        distance = self._distance()
        distance_change = self.previous_distance - distance

        # This is positive when approaching and automatically negative when
        # moving farther away
        reward = self.PROGRESS_REWARD_SCALE * distance_change - self.TIME_PENALTY
        terminated = distance <= self.ATTACK_RANGE
        if terminated and self.previous_distance > self.ATTACK_RANGE:
            reward += self.ATTACK_RANGE_REWARD

        truncated = self.elapsed_seconds >= self.EPISODE_SECONDS and not terminated
        self.previous_distance = distance

        info = self._get_info()
        info["distance_change"] = distance_change
        info["position_changed"] = bool(
            np.linalg.norm(self.bot_position - old_position) > 1e-6
        )
        info["success"] = terminated

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> str:

        grid = np.full((15, 15), ".", dtype="<U1")
        target_x, target_z = np.floor(self.target_position).astype(int)
        bot_x, bot_z = np.floor(self.bot_position).astype(int)
        grid[target_z, target_x] = "T"
        grid[bot_z, bot_x] = "B"
        return "\n".join("".join(row) for row in grid)

    def _apply_action(self, action: PositioningAction) -> None:
        turn_amount = self.TURN_SPEED_RADIANS * self.STEP_SECONDS
        if action == PositioningAction.TURN_LEFT:
            self.bot_yaw = self._wrap_angle(self.bot_yaw - turn_amount)
            return
        if action == PositioningAction.TURN_RIGHT:
            self.bot_yaw = self._wrap_angle(self.bot_yaw + turn_amount)
            return
        if action == PositioningAction.WAIT:
            return

        forward = np.array(
            [np.sin(self.bot_yaw), np.cos(self.bot_yaw)], dtype=np.float32
        )
        right = np.array([forward[1], -forward[0]], dtype=np.float32)
        direction = {
            PositioningAction.FORWARD: forward,
            PositioningAction.BACKWARD: -forward,
            PositioningAction.STRAFE_LEFT: -right,
            PositioningAction.STRAFE_RIGHT: right,
        }[action]

        movement = direction * self.MOVE_SPEED * self.STEP_SECONDS
        # Half-block margins model solid walls around the 15x15 arena.
        self.bot_position = np.clip(
            self.bot_position + movement, 0.5, self.ROOM_SIZE - 0.5
        ).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        relative_target = (self.target_position - self.bot_position) / self.ROOM_SIZE
        distance = self._distance()
        position_normalized = self.bot_position / (self.ROOM_SIZE / 2.0) - 1.0

        return np.array(
            [
                position_normalized[0],
                position_normalized[1],
                np.sin(self.bot_yaw),
                np.cos(self.bot_yaw),
                self.bot_health / self.MAX_HEALTH,
                float(self.has_equipment),
                relative_target[0],
                relative_target[1],
                distance / (np.sqrt(2.0) * self.ROOM_SIZE),
                float(distance <= self.ATTACK_RANGE),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "distance_to_target": self._distance(),
            "elapsed_seconds": self.elapsed_seconds,
            "health": self.bot_health,
            "has_equipment": self.has_equipment,
            "in_attack_range": self._distance() <= self.ATTACK_RANGE,
            "bot_position": self.bot_position.copy(),
            "target_position": self.target_position.copy(),
        }

    def _position_from_options(
        self, options: dict[str, Any], key: str
    ) -> np.ndarray:
        if key not in options:
            return self.np_random.uniform(
                0.5, self.ROOM_SIZE - 0.5, size=2
            ).astype(np.float32)

        position = np.asarray(options[key], dtype=np.float32)
        if position.shape != (2,):
            raise ValueError(f"{key} must contain exactly two coordinates: [x, z]")
        if np.any(position < 0.5) or np.any(position > self.ROOM_SIZE - 0.5):
            raise ValueError(f"{key} must be inside the 15x15 room")
        return position.copy()

    def _randomize_separated_positions(self) -> None:
        minimum_start_distance = self.ATTACK_RANGE + 2.0
        for _ in range(1_000):
            self.bot_position = self.np_random.uniform(
                0.5, self.ROOM_SIZE - 0.5, size=2
            ).astype(np.float32)
            self.target_position = self.np_random.uniform(
                0.5, self.ROOM_SIZE - 0.5, size=2
            ).astype(np.float32)
            if self._distance() >= minimum_start_distance:
                return
        raise RuntimeError("Could not generate valid starting positions")

    def _distance(self) -> float:
        return float(np.linalg.norm(self.target_position - self.bot_position))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


if __name__ == "__main__":
    env = BasicPositioningEnv(render_mode="ansi")
    observation, episode_info = env.reset(seed=7)
    print(env.render())
    print("\nInitial observation:", observation)
    print("Initial info:", episode_info)

    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        observation, reward, terminated, truncated, episode_info = env.step(
            env.action_space.sample()
        )
        total_reward += reward

    print(
        f"\nEpisode complete: success={episode_info['success']}, "
        f"seconds={episode_info['elapsed_seconds']:.1f}, "
        f"reward={total_reward:.3f}"
    )
