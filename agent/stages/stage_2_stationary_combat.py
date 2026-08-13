"""Stage two: defeat one stationary zombie in a hazard-free arena."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.minecraft_bridge import MinecraftWebSocketBridge


class StationaryCombatAction(IntEnum):
    WAIT = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    ATTACK = 7
    RETREAT = 8
    MOVE_TO_SAFE_AREA = 9


class StageTwoStationaryCombatEnv(gym.Env[np.ndarray, int]):
    """

    Observation:
        0-1. normalized bot x/z
        2-3. sin/cos of bot yaw
        4. bot health / 20
        5. whether the iron sword is equipped
        6-7. normalized zombie position relative to the bot
        8. normalized distance to the zombie
        9. whether the zombie is in attack range
        10. zombie health / 20
        11. whether the zombie is alive
        12. whether a fully charged sword attack is ready
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    ROOM_SIZE = 15.0
    MAX_HEALTH = 20.0
    ZOMBIE_MAX_HEALTH = 20.0
    # Use a safety margin below Minecraft's nominal three-block reach so the
    # observation and reward agree with server-side hit validation.
    ATTACK_RANGE = 2.5
    EPISODE_SECONDS = 60.0
    STEP_SECONDS = 0.1
    MOVE_SPEED = 4.3
    TURN_SPEED_RADIANS = np.pi
    IRON_SWORD_DAMAGE = 6.0
    ATTACK_COOLDOWN_SECONDS = 0.625

    DAMAGE_REWARD_SCALE = 1.0
    DEFEAT_REWARD = 20.0
    APPROACH_REWARD_SCALE = 1.0 # from 0.2 back to 1.0 to encourage closing distance more
    ATTACK_RANGE_ENTRY_REWARD = 0.5
    TIME_PENALTY = 0.01
    OUT_OF_RANGE_ATTACK_PENALTY = 0.10
    MINIMUM_START_DISTANCE = 5.0
    MAXIMUM_START_DISTANCE = 8.0

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.render_mode = render_mode
        # Stages two and three use the original eight actions. Later stages
        # may opt into the two extended movement actions without changing the
        # meaning of actions 0-7.
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0,
                    -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                dtype=np.float32,
            ),
            high=np.ones(13, dtype=np.float32),
            dtype=np.float32,
        )

        self.bot_position = np.zeros(2, dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        self.bot_yaw = 0.0
        self.bot_health = self.MAX_HEALTH
        self.bot_defeated = False
        self.has_iron_sword = True
        self.target_health = self.ZOMBIE_MAX_HEALTH
        self.target_alive = True
        self.elapsed_seconds = 0.0
        self.next_attack_time = 0.0
        self.attack_range_bonus_awarded = False
        self.target_visible = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        bot_option = options.get("bot_position")
        target_option = options.get("target_position")

        if bot_option is None and target_option is None:
            self._randomize_moderate_distance_positions()
        else:
            self.bot_position = self._validated_position(bot_option, "bot_position")
            self.target_position = self._validated_position(
                target_option, "target_position"
            )
            distance = self._distance()
            if not self.MINIMUM_START_DISTANCE <= distance <= self.MAXIMUM_START_DISTANCE:
                raise ValueError(
                    "Initial positions must be between 5 and 8 blocks apart"
                )

        self.bot_yaw = self._wrap_angle(
            float(options.get("bot_yaw", self.np_random.uniform(-np.pi, np.pi)))
        )
        self.bot_health = self.MAX_HEALTH
        self.bot_defeated = False
        self.has_iron_sword = True
        self.target_health = self.ZOMBIE_MAX_HEALTH
        self.target_alive = True
        self.elapsed_seconds = 0.0
        self.next_attack_time = 0.0
        self.attack_range_bonus_awarded = False
        self.target_visible = True
        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 7")

        selected_action = StationaryCombatAction(action)
        distance_before_action = self._distance()
        in_attack_range_before = self._is_in_attack_range()
        damage_dealt = 0.0
        attack_selected = selected_action == StationaryCombatAction.ATTACK
        invalid_attack = attack_selected and not in_attack_range_before
        valid_attack_attempt = False
        cooldown_blocked = False
        attack_landed = False

        if attack_selected:
            cooldown_blocked = (
                not invalid_attack and self.elapsed_seconds < self.next_attack_time
            )
            if (
                not invalid_attack
                and self.target_alive
                and not cooldown_blocked
            ):
                valid_attack_attempt = True
                damage_dealt = min(self.IRON_SWORD_DAMAGE, self.target_health)
                self.target_health -= damage_dealt
                self.target_alive = self.target_health > 0
                attack_landed = damage_dealt > 0
                self.next_attack_time = (
                    self.elapsed_seconds + self.ATTACK_COOLDOWN_SECONDS
                )
        else:
            self._apply_movement_action(selected_action)

        self.elapsed_seconds += self.STEP_SECONDS
        distance_after_action = self._distance()
        in_attack_range_after = self._is_in_attack_range()
        distance_change = distance_before_action - distance_after_action
        approach_reward = distance_change * self.APPROACH_REWARD_SCALE
        entered_attack_range = (
            not self.attack_range_bonus_awarded
            and not in_attack_range_before
            and in_attack_range_after
        )
        if entered_attack_range:
            self.attack_range_bonus_awarded = True

        reward = (
            damage_dealt * self.DAMAGE_REWARD_SCALE
            + approach_reward
            + (self.ATTACK_RANGE_ENTRY_REWARD if entered_attack_range else 0.0)
            - self.TIME_PENALTY
        )
        if invalid_attack:
            reward -= self.OUT_OF_RANGE_ATTACK_PENALTY
        if not self.target_alive:
            reward += self.DEFEAT_REWARD

        terminated = not self.target_alive
        truncated = self.elapsed_seconds >= self.EPISODE_SECONDS and not terminated
        info = self._get_info()
        info.update(
            {
                "damage_dealt": damage_dealt,
                "distance_change": distance_change,
                "approach_reward": approach_reward,
                "entered_attack_range": entered_attack_range,
                "attack_selected": attack_selected,
                "valid_attack_attempt": valid_attack_attempt,
                "invalid_attack": invalid_attack,
                "cooldown_blocked": cooldown_blocked,
                "attack_landed": attack_landed,
                "confirmed_kill": terminated,
                "attack_distance": (
                    distance_before_action if attack_selected else None
                ),
                "server_health_verified": False,
                "movement_settled": True,
                "movement_not_settled": False,
                "target_missing": False,
                "tracking_failure": False,
                "attack_packet_sent": valid_attack_attempt,
                "success": terminated,
                "source": "simulation",
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> str:
        grid = np.full((15, 15), ".", dtype="<U1")
        target_x, target_z = np.floor(self.target_position).astype(int)
        bot_x, bot_z = np.floor(self.bot_position).astype(int)
        if self.target_alive:
            grid[target_z, target_x] = "Z"
        grid[bot_z, bot_x] = "B"
        return "\n".join("".join(row) for row in grid)

    def _apply_movement_action(self, action: StationaryCombatAction) -> None:
        turn_amount = self.TURN_SPEED_RADIANS * self.STEP_SECONDS
        if action == StationaryCombatAction.TURN_LEFT:
            self.bot_yaw = self._wrap_angle(self.bot_yaw - turn_amount)
            return
        if action == StationaryCombatAction.TURN_RIGHT:
            self.bot_yaw = self._wrap_angle(self.bot_yaw + turn_amount)
            return
        if action == StationaryCombatAction.WAIT:
            return

        # Match Mineflayer/Minecraft yaw: yaw 0 faces -Z and -pi/2 faces +X.
        forward = np.array(
            [-np.sin(self.bot_yaw), -np.cos(self.bot_yaw)], dtype=np.float32
        )
        right = np.array([forward[1], -forward[0]], dtype=np.float32)
        direction = {
            StationaryCombatAction.FORWARD: forward,
            StationaryCombatAction.BACKWARD: -forward,
            StationaryCombatAction.STRAFE_LEFT: -right,
            StationaryCombatAction.STRAFE_RIGHT: right,
        }[action]
        movement = direction * self.MOVE_SPEED * self.STEP_SECONDS
        self.bot_position = np.clip(
            self.bot_position + movement, 0.5, self.ROOM_SIZE - 0.5
        ).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        relative_target = (self.target_position - self.bot_position) / self.ROOM_SIZE
        position_normalized = self.bot_position / (self.ROOM_SIZE / 2.0) - 1.0
        return np.array(
            [
                position_normalized[0],
                position_normalized[1],
                np.sin(self.bot_yaw),
                np.cos(self.bot_yaw),
                self.bot_health / self.MAX_HEALTH,
                float(self.has_iron_sword),
                relative_target[0],
                relative_target[1],
                self._distance() / (np.sqrt(2.0) * self.ROOM_SIZE),
                float(self._is_in_attack_range()),
                self.target_health / self.ZOMBIE_MAX_HEALTH,
                float(self.target_alive),
                float(self.elapsed_seconds >= self.next_attack_time),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "distance_to_target": self._distance(),
            "elapsed_seconds": self.elapsed_seconds,
            "bot_health": self.bot_health,
            "has_iron_sword": self.has_iron_sword,
            "target_health": self.target_health,
            "target_alive": self.target_alive,
            "target_visible": self.target_visible,
            "in_attack_range": self._is_in_attack_range(),
            "attack_ready": self.elapsed_seconds >= self.next_attack_time,
            "bot_position": self.bot_position.copy(),
            "target_position": self.target_position.copy(),
        }

    def _randomize_moderate_distance_positions(self) -> None:
        for _ in range(1_000):
            self.bot_position = self.np_random.uniform(
                0.5, self.ROOM_SIZE - 0.5, size=2
            ).astype(np.float32)
            self.target_position = self.np_random.uniform(
                0.5, self.ROOM_SIZE - 0.5, size=2
            ).astype(np.float32)
            if (
                self.MINIMUM_START_DISTANCE
                <= self._distance()
                <= self.MAXIMUM_START_DISTANCE
            ):
                return
        raise RuntimeError("Could not generate valid stage-two starting positions")

    def _validated_position(self, value: Any, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(
                "bot_position and target_position must be provided together"
            )
        position = np.asarray(value, dtype=np.float32)
        if position.shape != (2,):
            raise ValueError(f"{name} must contain exactly two coordinates: [x, z]")
        if np.any(position < 0.5) or np.any(position > self.ROOM_SIZE - 0.5):
            raise ValueError(f"{name} must be inside the 15x15 room")
        return position.copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self.target_position - self.bot_position))

    def _is_in_attack_range(self) -> bool:
        return self._distance() <= self.ATTACK_RANGE

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class LiveStageTwoStationaryCombatEnv(StageTwoStationaryCombatEnv):

    BRIDGE_STAGE = "stage2"
    BRIDGE_STAGE_NAME = "stage-two"

    def __init__(
        self,
        bridge_url: str = "ws://127.0.0.1:8765",
        *,
        bridge_timeout: float = 15.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.server_attack_distance: float | None = None
        self.server_in_attack_range = False
        self.bridge = MinecraftWebSocketBridge(bridge_url, timeout=bridge_timeout)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}
        super().reset(seed=seed, options=options)
        response = self.bridge.request(
            f"{self.BRIDGE_STAGE}.reset",
            botPosition=self.bot_position.tolist(),
            targetPosition=self.target_position.tolist(),
            botYaw=self.bot_yaw,
            botHealth=self.bot_health,
            rebuildArena=bool(options.get("rebuild_arena", False)),
            freezeTarget=bool(options.get("freeze_target", False)),
        )
        self._apply_live_state(response["state"])
        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 7")

        distance_before_action = self._distance()
        in_attack_range_before = self._is_in_attack_range()
        health_before_action = self.target_health
        response = self.bridge.request(
            f"{self.BRIDGE_STAGE}.step", action=int(action)
        )
        state = response["state"]
        attack_result = state["attackResult"]
        self._apply_live_state(response["state"])

        damage_dealt = max(0.0, health_before_action - self.target_health)
        distance_after_action = self._distance()
        in_attack_range_after = self._is_in_attack_range()
        distance_change = distance_before_action - distance_after_action
        approach_reward = distance_change * self.APPROACH_REWARD_SCALE
        entered_attack_range = (
            not self.attack_range_bonus_awarded
            and not in_attack_range_before
            and in_attack_range_after
        )
        if entered_attack_range:
            self.attack_range_bonus_awarded = True

        out_of_range = bool(attack_result["outOfRange"])
        target_missing = bool(attack_result.get("targetMissing", False))
        movement_not_settled = bool(
            attack_result.get("movementNotSettled", False)
        )
        tracking_failure = target_missing and self.target_alive
        reward = (
            damage_dealt * self.DAMAGE_REWARD_SCALE
            + approach_reward
            + (self.ATTACK_RANGE_ENTRY_REWARD if entered_attack_range else 0.0)
            - self.TIME_PENALTY
        )
        if out_of_range:
            reward -= self.OUT_OF_RANGE_ATTACK_PENALTY
        if tracking_failure:
            # A missing Mineflayer entity is an environment failure, not a
            # policy mistake. End the episode without rewarding or penalizing
            # the selected action.
            reward = 0.0

        terminated = not self.target_alive
        if terminated:
            reward += self.DEFEAT_REWARD
        truncated = (
            tracking_failure
            or self.elapsed_seconds >= self.EPISODE_SECONDS
        ) and not terminated
        info = self._get_info()
        info.update(
            {
                "damage_dealt": damage_dealt,
                "distance_change": distance_change,
                "approach_reward": approach_reward,
                "entered_attack_range": entered_attack_range,
                "attack_selected": bool(attack_result["attackSelected"]),
                "valid_attack_attempt": bool(
                    attack_result["validAttackAttempt"]
                ),
                "invalid_attack": out_of_range,
                "target_missing": target_missing,
                "tracking_failure": tracking_failure,
                "movement_not_settled": movement_not_settled,
                "cooldown_blocked": bool(attack_result["cooldownBlocked"]),
                "attack_landed": bool(attack_result["attackLanded"]),
                "confirmed_kill": bool(attack_result["confirmedKill"]),
                "attack_distance": attack_result["attackDistance"],
                "server_health_verified": bool(
                    attack_result["serverHealthVerified"]
                ),
                "movement_settled": bool(attack_result["movementSettled"]),
                "attack_packet_sent": bool(attack_result["attackPacketSent"]),
                "success": terminated,
                "source": "minecraft",
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.bridge.close()

    def _is_in_attack_range(self) -> bool:
        return self.server_in_attack_range

    def _apply_live_state(self, state: dict[str, Any]) -> None:
        try:
            bot_position = np.asarray(state["botPosition"], dtype=np.float32)
            target_position = np.asarray(state["targetPosition"], dtype=np.float32)
            if bot_position.shape != (2,) or target_position.shape != (2,):
                raise ValueError("positions must each contain [x, z]")

            self.bot_position = bot_position
            self.target_position = target_position
            self.bot_yaw = self._wrap_angle(float(state["yaw"]))
            self.bot_health = float(state["health"])
            self.bot_defeated = bool(state["botDefeated"])
            self.has_iron_sword = bool(state["hasIronSword"])
            self.target_health = float(state["targetHealth"])
            self.target_alive = bool(state["targetAlive"])
            self.target_visible = bool(state["targetVisible"])
            attack_distance = state["attackDistance"]
            self.server_attack_distance = (
                None if attack_distance is None else float(attack_distance)
            )
            self.server_in_attack_range = bool(state["inAttackRange"])
            self.elapsed_seconds = float(state["elapsedSeconds"])
            self.next_attack_time = (
                self.elapsed_seconds if state["attackReady"] else self.elapsed_seconds + 0.001
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid {self.BRIDGE_STAGE_NAME} state from Mineflayer: "
                f"{state!r}"
            ) from error
