"""Stage six: intercept a zombie and protect a stationary NPC."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.stages.stage_4_cover import (
    is_geometrically_occluded,
    point_inside_cover,
)
from agent.stages.stage_5_recovery import StageFiveRecoveryEnv


class StageSixProtectionEnv(StageFiveRecoveryEnv):
    DAMAGE_REWARD_SCALE = 0.5
    APPROACH_REWARD_SCALE = 0.5
    ATTACK_RANGE_ENTRY_REWARD = 0.25
    OUT_OF_RANGE_ATTACK_PENALTY = 0.05
    DAMAGE_TAKEN_PENALTY_SCALE = 0.25

    RETREAT_PROGRESS_REWARD_SCALE = 0.25
    COVER_PROGRESS_REWARD_SCALE = 0.1875
    COVER_PROGRESS_REWARD_CAP = 1.25
    COVER_ENTRY_REWARD = 0.5
    SAFE_EAT_REWARD = 2.0
    SUCCESSFUL_EAT_REWARD = 0.5
    HEALTH_RECOVERY_REWARD_SCALE = 0.375
    RECOVERY_COMPLETION_REWARD = 1.0
    REENGAGE_RANGE_REWARD = 0.25
    SURVIVAL_STEP_REWARD = 0.0025
    SURVIVED_TIMEOUT_REWARD = 2.5
    EAT_CLOSE_PENALTY = 1.0
    NO_FOOD_PENALTY = 0.25
    UNNECESSARY_EAT_PENALTY = 0.125
    PREMATURE_REENGAGE_PENALTY = 0.0625

    NPC_MAX_HEALTH = 20.0
    NPC_ZOMBIE_MIN_DISTANCE = 2.5
    NPC_ZOMBIE_MAX_DISTANCE = 5.0
    COVER_SPLIT_PROBABILITY = 0.35

    NPC_DAMAGE_PENALTY_SCALE = 3.0
    NPC_DEATH_PENALTY = 50.0
    NPC_SURVIVAL_REWARD = 30.0
    INTERACTION_RADIUS = 7.0
    STRAY_PENALTY_SCALE = 0.05

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        stage_five_low = self.observation_space.low.copy()
        stage_five_high = self.observation_space.high.copy()
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (stage_five_low, np.array([-1.0, -1.0, *([0.0] * 6)]))
            ).astype(np.float32),
            high=np.concatenate(
                (stage_five_high, np.ones(8, dtype=np.float32))
            ),
            dtype=np.float32,
        )
        self.npc_position = np.zeros(2, dtype=np.float32)
        self.npc_health = self.NPC_MAX_HEALTH
        self.npc_alive = True
        self.npc_under_attack = False
        self._npc_damage_accumulator = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        supplied = dict(options or {})
        gym.Env.reset(self, seed=seed)
        position_names = ("bot_position", "target_position", "npc_position")
        supplied_count = sum(name in supplied for name in position_names)
        if supplied_count not in (0, len(position_names)):
            raise ValueError(
                "bot_position, target_position, and npc_position must be "
                "provided together"
            )
        if supplied_count == 0:
            bot, zombie, npc = self._randomize_protection_positions()
            supplied.update(
                bot_position=bot.tolist(),
                target_position=zombie.tolist(),
                npc_position=npc.tolist(),
            )

        npc_position = self._validated_position(
            supplied["npc_position"], "npc_position"
        )
        if point_inside_cover(npc_position):
            raise ValueError("npc_position cannot be inside solid cover")
        if not (
            self.NPC_ZOMBIE_MIN_DISTANCE
            <= np.linalg.norm(
                np.asarray(supplied["target_position"]) - npc_position
            )
            <= self.NPC_ZOMBIE_MAX_DISTANCE
        ):
            raise ValueError("The zombie must start 2.5 to 5 blocks from the NPC")

        observation, info = super().reset(seed=seed, options=supplied)
        del observation
        self.bot_health = self.MAX_HEALTH
        self.starting_bot_health = self.MAX_HEALTH
        self.bot_defeated = False
        self.npc_position = npc_position.astype(np.float32)
        self.npc_health = self.NPC_MAX_HEALTH
        self.npc_alive = True
        self.npc_under_attack = self._npc_is_under_attack()
        self._npc_damage_accumulator = 0.0
        info.update(self._get_info())
        info.update(self._empty_protection_rewards())
        return self._get_observation(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._npc_damage_accumulator = 0.0
        observation, reward, terminated, truncated, info = super().step(action)
        del observation
        return self._apply_protection_rewards(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            npc_damage_taken=self._npc_damage_accumulator,
        )

    def _apply_protection_rewards(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        npc_damage_taken: float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward += float(info.get("premature_defeat_reward_removed", 0.0))
        reward -= float(info.get("recovery_victory_reward", 0.0))

        bot_damage_taken = float(info.get("damage_taken", 0.0))
        bot_damage_penalty = (
            bot_damage_taken * self.DAMAGE_TAKEN_PENALTY_SCALE
        )
        if info.get("source") == "minecraft":
            reward -= bot_damage_penalty

        npc_damage_penalty = (
            npc_damage_taken * self.NPC_DAMAGE_PENALTY_SCALE
        )
        reward -= npc_damage_penalty
        npc_defeated = not self.npc_alive or self.npc_health <= 0
        self.npc_alive = not npc_defeated
        npc_death_penalty = self.NPC_DEATH_PENALTY if npc_defeated else 0.0
        reward -= npc_death_penalty

        duration = int(info.get("action_duration_steps", 1))
        interaction_distance = self._interaction_distance()
        stray_penalty = (
            max(0.0, interaction_distance - self.INTERACTION_RADIUS)
            * self.STRAY_PENALTY_SCALE
            * duration
        )
        reward -= stray_penalty

        zombie_defeated = not self.target_alive
        terminated = terminated or npc_defeated
        truncated = (
            truncated or self.elapsed_seconds >= self.EPISODE_SECONDS
        ) and not terminated
        npc_survived_episode = self.npc_alive and (
            zombie_defeated or truncated
        )
        npc_survival_reward = (
            self.NPC_SURVIVAL_REWARD if npc_survived_episode else 0.0
        )
        reward += npc_survival_reward
        self.npc_under_attack = self._npc_is_under_attack()
        success = zombie_defeated and self.npc_alive

        info.update(self._get_info())
        info.update(
            {
                "npc_damage_taken": npc_damage_taken,
                "bot_damage_penalty": bot_damage_penalty,
                "npc_damage_penalty": npc_damage_penalty,
                "npc_defeated": npc_defeated,
                "npc_death_penalty": npc_death_penalty,
                "npc_survived_episode": npc_survived_episode,
                "npc_survival_reward": npc_survival_reward,
                "interaction_distance": interaction_distance,
                "stray_penalty": stray_penalty,
                "success": success,
                "scenario": "protect_npc",
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _advance_zombie(self) -> float:
        bot_distance = self._distance()
        bot_intercepts = (
            bot_distance
            <= self.ZOMBIE_ATTACK_RANGE + self.ZOMBIE_ATTACK_RANGE_EPSILON
            and not self._raw_cover_occlusion()
        )
        victim_position = (
            self.bot_position if bot_intercepts else self.npc_position
        )
        offset = victim_position - self.target_position
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
            candidate = self._cover_aware_zombie_candidate(movement)
            self.target_position = np.clip(
                candidate, 0.5, self.ROOM_SIZE - 0.5
            ).astype(np.float32)

        offset = victim_position - self.target_position
        distance = float(np.linalg.norm(offset))
        occluded = is_geometrically_occluded(
            victim_position, self.target_position
        )
        if (
            not occluded
            and distance
            <= self.ZOMBIE_ATTACK_RANGE + self.ZOMBIE_ATTACK_RANGE_EPSILON
            and self.elapsed_seconds >= self.next_zombie_attack_time
        ):
            self.next_zombie_attack_time = (
                self.elapsed_seconds + self.ZOMBIE_ATTACK_COOLDOWN_SECONDS
            )
            if bot_intercepts:
                damage = min(self.ZOMBIE_ATTACK_DAMAGE, self.bot_health)
                self.bot_health -= damage
                return float(damage)
            damage = min(self.ZOMBIE_ATTACK_DAMAGE, self.npc_health)
            self.npc_health -= damage
            self.npc_alive = self.npc_health > 0
            self._npc_damage_accumulator += float(damage)
        return 0.0

    def _cover_aware_zombie_candidate(
        self, movement: np.ndarray
    ) -> np.ndarray:
        candidate = self.target_position + movement
        if not point_inside_cover(candidate):
            return candidate
        alternatives = (
            self.target_position + np.array([movement[0], 0.0]),
            self.target_position + np.array([0.0, movement[1]]),
        )
        valid = [point for point in alternatives if not point_inside_cover(point)]
        if not valid:
            return self.target_position
        return min(
            valid,
            key=lambda point: float(np.linalg.norm(self.npc_position - point)),
        )

    def _get_observation(self) -> np.ndarray:
        stage_five = super()._get_observation()
        relative_npc = (self.npc_position - self.bot_position) / self.ROOM_SIZE
        diagonal = np.sqrt(2.0) * self.ROOM_SIZE
        protection = np.array(
            [
                relative_npc[0],
                relative_npc[1],
                self.npc_health / self.NPC_MAX_HEALTH,
                float(self.npc_alive),
                self._bot_npc_distance() / diagonal,
                self._enemy_npc_distance() / diagonal,
                float(self.npc_under_attack),
                self._interaction_distance() / diagonal,
            ],
            dtype=np.float32,
        )
        return np.concatenate((stage_five, protection)).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info.update(
            {
                "npc_position": self.npc_position.copy(),
                "npc_health": self.npc_health,
                "npc_alive": self.npc_alive,
                "npc_under_attack": self.npc_under_attack,
                "bot_npc_distance": self._bot_npc_distance(),
                "enemy_npc_distance": self._enemy_npc_distance(),
                "interaction_distance": self._interaction_distance(),
            }
        )
        return info

    def _bot_npc_distance(self) -> float:
        return float(np.linalg.norm(self.bot_position - self.npc_position))

    def _enemy_npc_distance(self) -> float:
        return float(np.linalg.norm(self.target_position - self.npc_position))

    def _interaction_distance(self) -> float:
        center = (self.target_position + self.npc_position) / 2.0
        return float(np.linalg.norm(self.bot_position - center))

    def _npc_is_under_attack(self) -> bool:
        return (
            self.npc_alive
            and self.target_alive
            and self._enemy_npc_distance()
            <= self.ZOMBIE_ATTACK_RANGE + 0.25
            and not is_geometrically_occluded(
                self.npc_position, self.target_position
            )
        )

    def _randomize_protection_positions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(1_000):
            if self.np_random.random() < self.COVER_SPLIT_PROBABILITY:
                zombie, npc = self._cover_split_positions()
            else:
                npc = self._random_open_point()
                angle = self.np_random.uniform(-np.pi, np.pi)
                distance = self.np_random.uniform(
                    self.NPC_ZOMBIE_MIN_DISTANCE,
                    self.NPC_ZOMBIE_MAX_DISTANCE,
                )
                zombie = npc + distance * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
            if not self._valid_open_point(zombie):
                continue
            bot = self._sample_bot_position(zombie, npc)
            if bot is not None:
                return bot, zombie.astype(np.float32), npc.astype(np.float32)
        raise RuntimeError("Could not generate valid Stage Six positions")

    def _cover_split_positions(self) -> tuple[np.ndarray, np.ndarray]:
        z_center = float(self.np_random.choice((3.5, 11.5)))
        z = z_center + self.np_random.uniform(-1.0, 1.0)
        npc_on_left = bool(self.np_random.integers(0, 2))
        left = np.array([5.5, z], dtype=np.float32)
        right = np.array([9.5, z], dtype=np.float32)
        return (right, left) if npc_on_left else (left, right)

    def _random_open_point(self) -> np.ndarray:
        for _ in range(100):
            point = self.np_random.uniform(1.0, self.ROOM_SIZE - 1.0, size=2)
            if self._valid_open_point(point):
                return point.astype(np.float32)
        raise RuntimeError("Could not sample an open arena position")

    def _sample_bot_position(
        self, zombie: np.ndarray, npc: np.ndarray
    ) -> np.ndarray | None:
        for _ in range(100):
            angle = self.np_random.uniform(-np.pi, np.pi)
            distance = self.np_random.uniform(
                self.MINIMUM_START_DISTANCE, self.MAXIMUM_START_DISTANCE
            )
            bot = zombie + distance * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            if (
                self._valid_open_point(bot)
                and np.linalg.norm(bot - npc) >= 3.0
            ):
                return bot.astype(np.float32)
        return None

    def _valid_open_point(self, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= 0.75)
            and np.all(point <= self.ROOM_SIZE - 0.75)
            and not point_inside_cover(point)
        )

    @staticmethod
    def _empty_protection_rewards() -> dict[str, float | bool]:
        return {
            "npc_damage_taken": 0.0,
            "bot_damage_penalty": 0.0,
            "npc_damage_penalty": 0.0,
            "npc_defeated": False,
            "npc_death_penalty": 0.0,
            "npc_survived_episode": False,
            "npc_survival_reward": 0.0,
            "stray_penalty": 0.0,
        }

    def render(self) -> str:
        grid = np.array([list(row) for row in super().render()])
        if self.npc_alive:
            npc_x, npc_z = np.floor(self.npc_position).astype(int)
            grid[npc_z, npc_x] = "P"
        return "\n".join("".join(row) for row in grid)
