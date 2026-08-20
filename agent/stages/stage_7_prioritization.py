"""Stage seven: prioritize two different threats while protecting an NPC."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agent.stages.stage_4_cover import (
    is_geometrically_occluded,
    point_inside_cover,
)
from agent.stages.stage_5_recovery import StageFiveAction
from agent.stages.stage_6_protection import StageSixProtectionEnv
from agent.stages.stage_7_threats import (
    THREAT_SPECS,
    THREAT_TYPES,
    StageSevenAction,
    threat_priority,
)


class StageSevenPrioritizationEnv(StageSixProtectionEnv):
    """Two-threat protection task with a hidden priority teacher."""
    # Stage Six shaping reduced by ten percent.
    DAMAGE_REWARD_SCALE = 0.45
    DEFEAT_REWARD = 18.0
    APPROACH_REWARD_SCALE = 0.45
    ATTACK_RANGE_ENTRY_REWARD = 0.225
    TIME_PENALTY = 0.009
    OUT_OF_RANGE_ATTACK_PENALTY = 0.045
    DAMAGE_TAKEN_PENALTY_SCALE = 0.225
    RETREAT_PROGRESS_REWARD_SCALE = 0.225
    COVER_PROGRESS_REWARD_SCALE = 0.16875
    COVER_PROGRESS_REWARD_CAP = 1.125
    COVER_ENTRY_REWARD = 0.45
    SAFE_EAT_REWARD = 1.8
    SUCCESSFUL_EAT_REWARD = 0.45
    HEALTH_RECOVERY_REWARD_SCALE = 0.3375
    RECOVERY_COMPLETION_REWARD = 0.9
    REENGAGE_RANGE_REWARD = 0.225
    SURVIVAL_STEP_REWARD = 0.00225
    SURVIVED_TIMEOUT_REWARD = 2.25
    EAT_CLOSE_PENALTY = 0.9
    NO_FOOD_PENALTY = 0.225
    UNNECESSARY_EAT_PENALTY = 0.1125
    PREMATURE_REENGAGE_PENALTY = 0.05625
    DEATH_PENALTY = 36.0
    NPC_DAMAGE_PENALTY_SCALE = 2.7
    NPC_DEATH_PENALTY = 45.0
    NPC_SURVIVAL_REWARD = 27.0
    STRAY_PENALTY_SCALE = 0.045

    PRIORITY_NEUTRALIZATION_REWARD = 8.0
    ENEMY_COUNT = 2
    ENEMY_NPC_MIN_DISTANCE = 2.5
    # The live arena inherits Stage Six's 2.5-to-5-block reset contract.
    ENEMY_NPC_MAX_DISTANCE = 5.0
    NPC_ZOMBIE_MAX_DISTANCE = ENEMY_NPC_MAX_DISTANCE
    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        stage_six_low = self.observation_space.low.copy()
        stage_six_high = self.observation_space.high.copy()
        threat_low = np.tile(
            np.array([-1.0, -1.0, *([0.0] * 12)], dtype=np.float32), 2
        )
        threat_high = np.ones(28, dtype=np.float32)
        self.action_space = spaces.Discrete(len(StageSevenAction))
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (stage_six_low, threat_low, np.zeros(2, dtype=np.float32))
            ),
            high=np.concatenate(
                (stage_six_high, threat_high, np.ones(2, dtype=np.float32))
            ),
            dtype=np.float32,
        )
        self.enemy_types = ["zombie", "skeleton"]
        self.enemy_positions = np.zeros((2, 2), dtype=np.float32)
        self.enemy_health = np.array([20.0, 20.0], dtype=np.float32)
        self.enemy_alive = np.ones(2, dtype=np.bool_)
        self.enemy_visible = np.ones(2, dtype=np.bool_)
        self.enemy_next_attack = np.zeros(2, dtype=np.float32)
        self.selected_enemy = 0
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        supplied = dict(options or {})
        gym.Env.reset(self, seed=seed)
        enemy_types = supplied.pop("enemy_types", None)
        if enemy_types is None:
            enemy_types = self.np_random.choice(
                THREAT_TYPES, size=2, replace=False
            ).tolist()
        self.enemy_types = self._validate_enemy_types(enemy_types)

        position_keys = ("bot_position", "enemy_positions", "npc_position")
        supplied_count = sum(key in supplied for key in position_keys)
        if supplied_count not in (0, len(position_keys)):
            raise ValueError(
                "bot_position, enemy_positions, and npc_position must be "
                "provided together"
            )
        if supplied_count == 0:
            bot, enemies, npc = self._randomize_stage_seven_positions()
        else:
            bot = self._validated_position(
                supplied.pop("bot_position"), "bot_position"
            )
            npc = self._validated_position(
                supplied.pop("npc_position"), "npc_position"
            )
            enemies = self._validate_enemy_positions(
                supplied.pop("enemy_positions"), npc
            )

        self.enemy_positions = enemies.astype(np.float32)
        self.enemy_health = np.array(
            [THREAT_SPECS[name].max_health for name in self.enemy_types],
            dtype=np.float32,
        )
        self.enemy_alive[:] = True
        self.enemy_visible[:] = True
        self.enemy_next_attack[:] = 0.0
        self.selected_enemy = 0
        supplied.update(
            bot_position=bot.tolist(),
            target_position=enemies[0].tolist(),
            npc_position=npc.tolist(),
            bot_health=self.MAX_HEALTH,
        )
        _, info = super().reset(seed=seed, options=supplied)
        self._sync_selected_to_base()
        info.update(self._get_info())
        info.update(self._empty_priority_rewards())
        return self._get_observation(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Invalid Stage Seven action; expected 0 to 13")
        priority_before = self._highest_priority_enemy()
        alive_before = self.enemy_alive.copy()

        if action in (
            int(StageSevenAction.SELECT_ENEMY_1),
            int(StageSevenAction.SELECT_ENEMY_2),
        ):
            requested = action - int(StageSevenAction.SELECT_ENEMY_1)
            if self.enemy_alive[requested]:
                self.selected_enemy = requested
            self._sync_selected_to_base()
            result = StageSixProtectionEnv.step(self, int(StageFiveAction.WAIT))
        else:
            self._sync_selected_to_base()
            result = StageSixProtectionEnv.step(self, action)
        self._store_selected_from_base()
        return self._finalize_stage_seven(
            *result,
            action=action,
            priority_before=priority_before,
            alive_before=alive_before,
        )

    def _finalize_stage_seven(
        self,
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        *,
        action: int,
        priority_before: int,
        alive_before: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del observation
        newly_defeated = np.flatnonzero(alive_before & ~self.enemy_alive)
        neutralized = int(newly_defeated[0]) if len(newly_defeated) else -1
        all_defeated = not bool(np.any(self.enemy_alive))

        stage_six_survival = float(info.get("npc_survival_reward", 0.0))
        if stage_six_survival and not all_defeated and not truncated:
            reward -= stage_six_survival
            info["npc_survival_reward"] = 0.0
            info["npc_survived_episode"] = False

        priority_reward = (
            self.PRIORITY_NEUTRALIZATION_REWARD
            if neutralized == priority_before
            else 0.0
        )
        reward += priority_reward
        npc_defeated = bool(info.get("npc_defeated", False))
        bot_defeated = bool(info.get("bot_defeated", False))
        terminated = npc_defeated or bot_defeated or all_defeated
        truncated = (
            self.elapsed_seconds >= self.EPISODE_SECONDS and not terminated
        )

        if not self.enemy_alive[self.selected_enemy] and not all_defeated:
            self.selected_enemy = int(np.flatnonzero(self.enemy_alive)[0])
        self._sync_selected_to_base()
        success = all_defeated and self.npc_alive and not bot_defeated
        info.update(self._get_info())
        info.update(
            {
                "selected_action": action,
                "highest_priority_enemy": priority_before,
                "neutralized_enemy": neutralized,
                "neutralized_highest_priority": neutralized == priority_before,
                "priority_neutralization_reward": priority_reward,
                "all_enemies_defeated": all_defeated,
                "success": success,
                "scenario": "prioritize_two_threats",
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _advance_zombie(self) -> float:
        self._store_selected_from_base()
        bot_damage = 0.0
        for index in range(self.ENEMY_COUNT):
            if not self.enemy_alive[index]:
                continue
            spec = THREAT_SPECS[self.enemy_types[index]]
            position = self.enemy_positions[index]
            bot_distance = float(np.linalg.norm(self.bot_position - position))
            bot_intercepts = (
                bot_distance <= spec.attack_range + 1e-4
                and not is_geometrically_occluded(self.bot_position, position)
            )
            victim = self.bot_position if bot_intercepts else self.npc_position
            offset = victim - position
            distance = float(np.linalg.norm(offset))
            occluded = is_geometrically_occluded(position, victim)
            if (distance > spec.attack_range or occluded) and distance > 0:
                movement_length = min(
                    spec.move_speed * self.STEP_SECONDS,
                    max(0.0, distance - min(spec.attack_range, 1.8)),
                )
                movement = offset / distance * movement_length
                self.enemy_positions[index] = self._cover_aware_enemy_candidate(
                    position, movement, victim
                )
                position = self.enemy_positions[index]
                distance = float(np.linalg.norm(victim - position))
                occluded = is_geometrically_occluded(position, victim)

            if (
                not occluded
                and distance <= spec.attack_range + 1e-4
                and self.elapsed_seconds >= self.enemy_next_attack[index]
            ):
                self.enemy_next_attack[index] = (
                    self.elapsed_seconds + spec.cooldown_seconds
                )
                if bot_intercepts:
                    damage = min(spec.attack_damage, self.bot_health)
                    self.bot_health -= damage
                    bot_damage += damage
                else:
                    damage = min(spec.attack_damage, self.npc_health)
                    self.npc_health -= damage
                    self.npc_alive = self.npc_health > 0
                    self._npc_damage_accumulator += float(damage)
        self._sync_selected_to_base()
        return float(bot_damage)

    def _cover_aware_enemy_candidate(
        self, position: np.ndarray, movement: np.ndarray, victim: np.ndarray
    ) -> np.ndarray:
        candidate = position + movement
        if not point_inside_cover(candidate):
            return np.clip(candidate, 0.5, self.ROOM_SIZE - 0.5).astype(np.float32)
        alternatives = (
            position + np.array([movement[0], 0.0]),
            position + np.array([0.0, movement[1]]),
        )
        valid = [point for point in alternatives if not point_inside_cover(point)]
        if not valid:
            return position
        return min(valid, key=lambda point: float(np.linalg.norm(victim - point))).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        stage_six = super()._get_observation()
        if not hasattr(self, "enemy_positions"):
            return np.concatenate((stage_six, np.zeros(30, dtype=np.float32)))
        selected_spec = THREAT_SPECS[self.enemy_types[self.selected_enemy]]
        stage_six[10] = self.target_health / selected_spec.max_health
        diagonal = np.sqrt(2.0) * self.ROOM_SIZE
        values: list[float] = []
        for index, enemy_type in enumerate(self.enemy_types):
            spec = THREAT_SPECS[enemy_type]
            relative = (self.enemy_positions[index] - self.bot_position) / self.ROOM_SIZE
            attacking_npc, attacking_bot = self._attack_focus(index)
            values.extend(
                [
                    float(relative[0]),
                    float(relative[1]),
                    float(self.enemy_health[index] / spec.max_health),
                    float(self.enemy_alive[index]),
                    self._slot_bot_distance(index) / diagonal,
                    self._slot_npc_distance(index) / diagonal,
                    float(spec.ranged),
                    float(attacking_npc),
                    float(attacking_bot),
                    *(float(enemy_type == name) for name in THREAT_TYPES),
                ]
            )
        selected = [float(self.selected_enemy == index) for index in range(2)]
        return np.concatenate(
            (stage_six, np.asarray(values + selected, dtype=np.float32))
        ).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        if not hasattr(self, "enemy_positions"):
            return info
        priorities = self._priority_scores()
        info.update(
            {
                "enemy_types": tuple(self.enemy_types),
                "enemy_positions": self.enemy_positions.copy(),
                "enemy_health": self.enemy_health.copy(),
                "enemy_max_health": np.array(
                    [THREAT_SPECS[name].max_health for name in self.enemy_types]
                ),
                "enemy_alive": self.enemy_alive.copy(),
                "enemy_visible": self.enemy_visible.copy(),
                "enemy_bot_distances": np.array(
                    [self._slot_bot_distance(i) for i in range(2)]
                ),
                "enemy_npc_distances": np.array(
                    [self._slot_npc_distance(i) for i in range(2)]
                ),
                "enemy_attacking_npc": np.array(
                    [self._attack_focus(i)[0] for i in range(2)]
                ),
                "enemy_attacking_bot": np.array(
                    [self._attack_focus(i)[1] for i in range(2)]
                ),
                "selected_enemy": self.selected_enemy,
                "threat_priorities": priorities,
                "highest_priority_enemy": int(np.argmax(priorities)),
            }
        )
        return info

    def _priority_scores(self) -> np.ndarray:
        diagonal = np.sqrt(2.0) * self.ROOM_SIZE
        scores = []
        for index, enemy_type in enumerate(self.enemy_types):
            if not self.enemy_alive[index]:
                scores.append(float("-inf"))
                continue
            spec = THREAT_SPECS[enemy_type]
            attacking_npc, attacking_bot = self._attack_focus(index)
            scores.append(
                threat_priority(
                    bot_distance=self._slot_bot_distance(index),
                    npc_distance=self._slot_npc_distance(index),
                    health=float(self.enemy_health[index]),
                    max_health=spec.max_health,
                    ranged=spec.ranged,
                    attacking_npc=attacking_npc,
                    attacking_bot=attacking_bot,
                    arena_diagonal=diagonal,
                )
            )
        return np.asarray(scores, dtype=np.float32)

    def _highest_priority_enemy(self) -> int:
        return int(np.argmax(self._priority_scores()))
    def _attack_focus(self, index: int) -> tuple[bool, bool]:
        if not self.enemy_alive[index]:
            return False, False
        position = self.enemy_positions[index]
        spec = THREAT_SPECS[self.enemy_types[index]]
        bot_distance = self._slot_bot_distance(index)
        bot_attack = (
            bot_distance <= spec.attack_range + 0.25
            and not is_geometrically_occluded(position, self.bot_position)
        )
        npc_attack = (
            not bot_attack
            and self._slot_npc_distance(index) <= spec.attack_range + 0.25
            and not is_geometrically_occluded(position, self.npc_position)
        )
        return npc_attack, bot_attack

    def _sync_selected_to_base(self) -> None:
        index = self.selected_enemy
        self.target_position = self.enemy_positions[index].copy()
        self.target_health = float(self.enemy_health[index])
        self.target_alive = bool(self.enemy_alive[index])
        self.target_visible = bool(self.enemy_visible[index])
        self.next_zombie_attack_time = float(self.enemy_next_attack[index])

    def _store_selected_from_base(self) -> None:
        index = self.selected_enemy
        self.enemy_positions[index] = self.target_position.copy()
        self.enemy_health[index] = self.target_health
        self.enemy_alive[index] = self.target_alive
        self.enemy_visible[index] = self.target_visible
        self.enemy_next_attack[index] = self.next_zombie_attack_time

    def _slot_bot_distance(self, index: int) -> float:
        return float(np.linalg.norm(self.enemy_positions[index] - self.bot_position))

    def _slot_npc_distance(self, index: int) -> float:
        return float(np.linalg.norm(self.enemy_positions[index] - self.npc_position))

    def _interaction_distance(self) -> float:
        if not hasattr(self, "enemy_positions"):
            return super()._interaction_distance()
        alive = self.enemy_positions[self.enemy_alive]
        enemy_center = (
            np.mean(alive, axis=0) if len(alive) else self.npc_position
        )
        center = (enemy_center + self.npc_position) / 2.0
        return float(np.linalg.norm(self.bot_position - center))

    def _npc_is_under_attack(self) -> bool:
        if not hasattr(self, "enemy_positions"):
            return super()._npc_is_under_attack()
        return any(self._attack_focus(index)[0] for index in range(2))

    def _randomize_stage_seven_positions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(2_000):
            npc = self._random_open_point()
            enemies = []
            for _index in range(2):
                angle = self.np_random.uniform(-np.pi, np.pi)
                distance = self.np_random.uniform(
                    self.ENEMY_NPC_MIN_DISTANCE,
                    self.ENEMY_NPC_MAX_DISTANCE,
                )
                enemy = npc + distance * np.array([np.cos(angle), np.sin(angle)])
                enemies.append(enemy)
            enemy_array = np.asarray(enemies, dtype=np.float32)
            if not all(self._valid_open_point(enemy) for enemy in enemy_array):
                continue
            if np.linalg.norm(enemy_array[0] - enemy_array[1]) < 2.0:
                continue
            bot = self._sample_bot_position(enemy_array[0], npc)
            if bot is not None and np.linalg.norm(bot - enemy_array[1]) >= 2.5:
                return bot, enemy_array, npc
        raise RuntimeError("Could not generate valid Stage Seven positions")

    def _validate_enemy_positions(
        self, value: Any, npc: np.ndarray
    ) -> np.ndarray:
        positions = np.asarray(value, dtype=np.float32)
        if positions.shape != (2, 2):
            raise ValueError("enemy_positions must contain two [x, z] positions")
        for position in positions:
            if not self._valid_open_point(position):
                raise ValueError("enemy positions must be open points in the arena")
            distance = float(np.linalg.norm(position - npc))
            if not self.ENEMY_NPC_MIN_DISTANCE <= distance <= self.ENEMY_NPC_MAX_DISTANCE:
                raise ValueError("each enemy must start 2.5 to 5 blocks from the NPC")
        if np.linalg.norm(positions[0] - positions[1]) < 2.0:
            raise ValueError("the two enemies must start at least 2 blocks apart")
        return positions.copy()

    @staticmethod
    def _validate_enemy_types(value: Any) -> list[str]:
        names = [str(name).lower() for name in value]
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("enemy_types must contain two different enemies")
        if any(name not in THREAT_TYPES for name in names):
            raise ValueError(f"enemy_types must come from {THREAT_TYPES}")
        return names

    @staticmethod
    def _empty_priority_rewards() -> dict[str, float | bool | int]:
        return {
            "neutralized_enemy": -1,
            "neutralized_highest_priority": False,
            "priority_neutralization_reward": 0.0,
            "all_enemies_defeated": False,
        }

    def render(self) -> str:
        grid = np.array([list(row) for row in super().render()])
        symbols = {"zombie": "Z", "skeleton": "K", "enderman": "E", "spider": "S", "witch": "W"}
        for index, enemy_type in enumerate(self.enemy_types):
            if self.enemy_alive[index]:
                x, z = np.floor(self.enemy_positions[index]).astype(int)
                grid[z, x] = symbols[enemy_type]
        return "\n".join("".join(row) for row in grid)
