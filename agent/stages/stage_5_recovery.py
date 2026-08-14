"""Stage five: retreat, recover with food, then re-engage a zombie."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from gymnasium import spaces

from agent.stages.stage_2_stationary_combat import StationaryCombatAction
from agent.stages.stage_3_moving_combat import StageThreeMovingCombatEnv
from agent.stages.stage_4_retreat import StageFourRetreatEnv
from agent.stages.stage_4_rewards import bounded_positive_reward


class StageFiveAction(IntEnum):
    """Stage Four actions plus recovery and directed re-engagement."""

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
    EAT = 10
    REENGAGE = 11


class StageFiveRecoveryEnv(StageFourRetreatEnv):

    STARTING_HEALTH = 15.0
    DEFAULT_FOOD_COUNT = 64
    MAX_FOOD_COUNT = 64
    HEALTH_PER_FOOD = 5.0
    RECOVERED_HEALTH_THRESHOLD = 19.0

    EAT_DURATION_STEPS = 16
    REENGAGE_COMMITMENT_STEPS = 5
    SAFE_EAT_DISTANCE = 4.0

    SAFE_EAT_REWARD = 8.0
    SUCCESSFUL_EAT_REWARD = 2.0
    HEALTH_RECOVERY_REWARD_SCALE = 1.5
    RECOVERY_COMPLETION_REWARD = 4.0
    REENGAGE_RANGE_REWARD = 1.0
    DEFEAT_AFTER_RECOVERY_REWARD = 25.0
    SURVIVAL_STEP_REWARD = 0.01
    SURVIVED_TIMEOUT_REWARD = 10.0

    EAT_CLOSE_PENALTY = 4.0
    UNSAFE_EAT_PENALTY = 0.0
    NO_FOOD_PENALTY = 1.0
    UNNECESSARY_EAT_PENALTY = 0.5
    PREMATURE_REENGAGE_PENALTY = 0.25
    DEATH_PENALTY = 40.0

    RETREAT_PROGRESS_REWARD_SCALE = 1.0
    COVER_PROGRESS_REWARD_SCALE = 0.75
    COVER_PROGRESS_REWARD_CAP = 5.0
    COVER_ENTRY_REWARD = 2.0

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        stage_four_low = self.observation_space.low.copy()
        stage_four_high = self.observation_space.high.copy()
        self.action_space = spaces.Discrete(len(StageFiveAction))
        self.observation_space = spaces.Box(
            low=np.concatenate((stage_four_low, np.zeros(4, dtype=np.float32))),
            high=np.concatenate((stage_four_high, np.ones(4, dtype=np.float32))),
            dtype=np.float32,
        )
        self.food_count = self.DEFAULT_FOOD_COUNT
        self.has_eaten = False
        self.has_recovered = False
        self.reengaged_after_recovery = False
        self.stage_five_cover_progress_earned = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        supplied_options = dict(options or {})
        supplied_options["bot_health"] = self.STARTING_HEALTH
        food_count = int(
            supplied_options.pop("food_count", self.DEFAULT_FOOD_COUNT)
        )
        if not 1 <= food_count <= self.MAX_FOOD_COUNT:
            raise ValueError(
                f"food_count must be between 1 and {self.MAX_FOOD_COUNT}"
            )

        observation, info = StageThreeMovingCombatEnv.reset(
            self,
            seed=seed,
            options=supplied_options,
        )
        del observation
        self.starting_bot_health = self.STARTING_HEALTH
        self.survival_mode = False
        self.bot_health = self.STARTING_HEALTH
        self.bot_defeated = False
        self.cover_bonus_awarded = False
        self.cover_progress_reward_earned = 0.0
        self.cover_streak = 0
        self.confirmed_in_cover = False
        self.food_count = food_count
        self.has_eaten = False
        self.has_recovered = False
        self.reengaged_after_recovery = False
        self.stage_five_cover_progress_earned = 0.0
        info.update(self._get_info())
        info.update(self._empty_stage_five_rewards())
        return self._get_observation(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid Stage Five action {action}; expected 0 to "
                f"{self.action_space.n - 1}"
            )

        selected = StageFiveAction(action)
        health_before = self.bot_health
        food_count_before = self.food_count
        recovered_before = self.has_recovered
        target_distance_before = self._distance()
        safe_distance_before = self._safe_distance()
        in_cover_before = self.confirmed_in_cover
        was_in_attack_range = self._is_in_attack_range()

        if selected == StageFiveAction.EAT:
            reward, terminated, truncated, info, duration_steps = (
                self._run_base_action(
                    StationaryCombatAction.WAIT,
                    self.EAT_DURATION_STEPS,
                )
            )
        elif selected == StageFiveAction.REENGAGE:
            reward, terminated, truncated, info, duration_steps = (
                self._run_reengage_action()
            )
        elif selected == StageFiveAction.MOVE_TO_SAFE_AREA:
            reward, terminated, truncated, info, duration_steps = (
                self._run_base_action(
                    StationaryCombatAction.MOVE_TO_SAFE_AREA,
                    self.SAFE_AREA_COMMITMENT_STEPS,
                )
            )
        else:
            reward, terminated, truncated, info, duration_steps = (
                self._run_base_action(StationaryCombatAction(action), 1)
            )

        return self._apply_stage_five_rewards(
            selected=selected,
            health_before=health_before,
            food_count_before=food_count_before,
            recovered_before=recovered_before,
            target_distance_before=target_distance_before,
            safe_distance_before=safe_distance_before,
            in_cover_before=in_cover_before,
            was_in_attack_range=was_in_attack_range,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            duration_steps=duration_steps,
            source="simulation",
        )

    def _run_base_action(
        self,
        action: StationaryCombatAction,
        maximum_steps: int,
    ) -> tuple[float, bool, bool, dict[str, Any], int]:
        reward = 0.0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        totals = {
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "distance_change": 0.0,
            "approach_reward": 0.0,
        }
        flags = {
            "entered_attack_range": False,
            "attack_selected": False,
            "valid_attack_attempt": False,
            "invalid_attack": False,
            "attack_landed": False,
        }

        for step_index in range(maximum_steps):
            _, step_reward, terminated, truncated, last_info = (
                StageThreeMovingCombatEnv.step(self, int(action))
            )
            reward += step_reward
            for name in totals:
                totals[name] += float(last_info[name])
            for name in flags:
                flags[name] = flags[name] or bool(last_info[name])
            self._update_cover_confirmation()
            if terminated or truncated:
                break
            if (
                action == StationaryCombatAction.MOVE_TO_SAFE_AREA
                and self.confirmed_in_cover
            ):
                break

        last_info.update(totals)
        last_info.update(flags)
        return reward, terminated, truncated, last_info, step_index + 1

    def _run_reengage_action(
        self,
    ) -> tuple[float, bool, bool, dict[str, Any], int]:
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        executed_steps = 0
        for _ in range(self.REENGAGE_COMMITMENT_STEPS):
            direction = self.target_position - self.bot_position
            self.bot_yaw = self._wrap_angle(
                float(np.arctan2(-direction[0], -direction[1]))
            )
            step_reward, terminated, truncated, info, _ = (
                self._run_base_action(StationaryCombatAction.FORWARD, 1)
            )
            reward += step_reward
            executed_steps += 1
            if terminated or truncated or self._is_in_attack_range():
                break
        return reward, terminated, truncated, info, executed_steps

    def _apply_stage_five_rewards(
        self,
        *,
        selected: StageFiveAction,
        health_before: float,
        food_count_before: int,
        recovered_before: bool,
        target_distance_before: float,
        safe_distance_before: float,
        in_cover_before: bool,
        was_in_attack_range: bool,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        duration_steps: int,
        source: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_distance_change = target_distance_before - self._distance()
        safe_distance_change = safe_distance_before - self._safe_distance()
        alive = not bool(info.get("bot_defeated", False))
        safe_when_eating_started = (
            in_cover_before
            and target_distance_before >= self.SAFE_EAT_DISTANCE
        )

        retreat_reward = 0.0
        if selected in (
            StageFiveAction.RETREAT,
            StageFiveAction.MOVE_TO_SAFE_AREA,
            StageFiveAction.EAT,
        ):
            reward -= float(info["approach_reward"])
        if selected == StageFiveAction.RETREAT and self.target_alive:
            retreat_reward = (
                -target_distance_change * self.RETREAT_PROGRESS_REWARD_SCALE
            )
            reward += retreat_reward

        cover_progress_reward = 0.0
        if (
            selected == StageFiveAction.MOVE_TO_SAFE_AREA
            and not in_cover_before
            and self.target_alive
        ):
            raw_cover_reward = (
                safe_distance_change * self.COVER_PROGRESS_REWARD_SCALE
            )
            (
                cover_progress_reward,
                self.stage_five_cover_progress_earned,
            ) = bounded_positive_reward(
                raw_cover_reward,
                self.stage_five_cover_progress_earned,
                self.COVER_PROGRESS_REWARD_CAP,
            )
            reward += cover_progress_reward
        if (
            not self.cover_bonus_awarded
            and self.confirmed_in_cover
            and self.target_alive
        ):
            cover_progress_reward += self.COVER_ENTRY_REWARD
            reward += self.COVER_ENTRY_REWARD
            self.cover_bonus_awarded = True

        safe_eat_reward = 0.0
        successful_eat_reward = 0.0
        recovered_health = (
            float(info.get("server_recovered_health", 0.0))
            if source == "minecraft"
            else 0.0
        )
        recovery_reward = (
            recovered_health * self.HEALTH_RECOVERY_REWARD_SCALE
        )
        recovery_completion_reward = 0.0
        eating_close_penalty = 0.0
        unsafe_eat_penalty = 0.0
        no_food_penalty = 0.0
        unnecessary_eat_penalty = 0.0
        ate_successfully = bool(info.get("ate_successfully", False))
        if selected == StageFiveAction.EAT and alive:
            if food_count_before <= 0 or bool(info.get("eat_no_food", False)):
                no_food_penalty = self.NO_FOOD_PENALTY
            elif health_before >= self.MAX_HEALTH:
                unnecessary_eat_penalty = self.UNNECESSARY_EAT_PENALTY
            else:
                if target_distance_before < self.SAFE_EAT_DISTANCE:
                    eating_close_penalty = self.EAT_CLOSE_PENALTY
                elif not safe_when_eating_started:
                    unsafe_eat_penalty = self.UNSAFE_EAT_PENALTY
                if source == "simulation":
                    self.food_count -= 1
                    recovered_health = min(
                        self.HEALTH_PER_FOOD,
                        self.MAX_HEALTH - self.bot_health,
                    )
                    self.bot_health += recovered_health
                    self.has_eaten = True
                    ate_successfully = True
                    recovery_reward = (
                        recovered_health * self.HEALTH_RECOVERY_REWARD_SCALE
                    )
                if ate_successfully and safe_when_eating_started:
                    safe_eat_reward = self.SAFE_EAT_REWARD
                if ate_successfully:
                    successful_eat_reward = self.SUCCESSFUL_EAT_REWARD
                if (
                    source == "simulation"
                    and self.bot_health >= self.RECOVERED_HEALTH_THRESHOLD
                ):
                    self.has_recovered = True
            reward += (
                safe_eat_reward
                + successful_eat_reward
                + recovery_reward
            )
            reward -= eating_close_penalty + unsafe_eat_penalty
            reward -= no_food_penalty + unnecessary_eat_penalty

        if not recovered_before and self.has_recovered:
            recovery_completion_reward = self.RECOVERY_COMPLETION_REWARD
        reward += recovery_completion_reward
        if source == "minecraft" and selected != StageFiveAction.EAT:
            reward += recovery_reward

        reengage_reward = 0.0
        premature_reengage_penalty = 0.0
        if selected == StageFiveAction.REENGAGE:
            if self.has_recovered:
                entered_range = (
                    not was_in_attack_range and self._is_in_attack_range()
                )
                if entered_range:
                    reengage_reward = self.REENGAGE_RANGE_REWARD
                    reward += reengage_reward
                if self._is_in_attack_range():
                    self.reengaged_after_recovery = True
            else:
                premature_reengage_penalty = (
                    self.PREMATURE_REENGAGE_PENALTY
                )
                reward -= premature_reengage_penalty

        bot_defeated = bool(info.get("bot_defeated", False))
        death_penalty = self.DEATH_PENALTY if bot_defeated else 0.0
        reward -= death_penalty

        survival_reward = 0.0
        if alive:
            survival_reward = self.SURVIVAL_STEP_REWARD * duration_steps
            reward += survival_reward

        survived_timeout = truncated and alive and self.target_alive
        timeout_reward = (
            self.SURVIVED_TIMEOUT_REWARD if survived_timeout else 0.0
        )
        reward += timeout_reward

        defeated_after_recovery = not self.target_alive and self.has_recovered
        premature_defeat_reward_removed = 0.0
        if not self.target_alive and not self.has_recovered:
            # The inherited combat layer rewards every kill. Stage Five's
            # objective specifically requires recovery before victory, so a
            # premature kill must not receive that terminal bonus.
            premature_defeat_reward_removed = self.DEFEAT_REWARD
            reward -= premature_defeat_reward_removed
        recovery_victory_reward = (
            self.DEFEAT_AFTER_RECOVERY_REWARD
            if defeated_after_recovery
            else 0.0
        )
        reward += recovery_victory_reward

        info.update(self._get_info())
        info.update(
            {
                "damage_dealt": float(info["damage_dealt"]),
                "damage_taken": float(info["damage_taken"]),
                "retreat_progress_reward": retreat_reward,
                "cover_progress_reward": cover_progress_reward,
                "safe_eat_reward": safe_eat_reward,
                "successful_eat_reward": successful_eat_reward,
                "recovered_health": recovered_health,
                "health_recovery_reward": recovery_reward,
                "recovery_completion_reward": recovery_completion_reward,
                "eating_close_penalty": eating_close_penalty,
                "unsafe_eat_penalty": unsafe_eat_penalty,
                "no_food_penalty": no_food_penalty,
                "unnecessary_eat_penalty": unnecessary_eat_penalty,
                "ate_successfully": ate_successfully,
                "reengage_reward": reengage_reward,
                "premature_reengage_penalty": premature_reengage_penalty,
                "survival_reward": survival_reward,
                "survived_timeout_reward": timeout_reward,
                "death_penalty": death_penalty,
                "defeated_after_recovery": defeated_after_recovery,
                "premature_defeat_reward_removed": (
                    premature_defeat_reward_removed
                ),
                "recovery_victory_reward": recovery_victory_reward,
                "action_duration_steps": duration_steps,
                "success": defeated_after_recovery,
                "source": source,
            }
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        stage_four = super()._get_observation()
        stage_five = np.array(
            [
                self.food_count / self.MAX_FOOD_COUNT,
                float(self.has_eaten),
                float(self.has_recovered),
                float(self._is_safe_to_eat()),
            ],
            dtype=np.float32,
        )
        return np.concatenate((stage_four, stage_five)).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info.update(
            {
                "food_count": self.food_count,
                "has_eaten": self.has_eaten,
                "has_recovered": self.has_recovered,
                "safe_to_eat": self._is_safe_to_eat(),
                "reengaged_after_recovery": self.reengaged_after_recovery,
                "scenario": "retreat_eat_reengage",
            }
        )
        return info

    def _is_safe_to_eat(self) -> bool:
        return (
            self.confirmed_in_cover
            and self._distance() >= self.SAFE_EAT_DISTANCE
        )

    def _safe_distance(self) -> float:
        return float(
            np.linalg.norm(self._nearest_safe_position() - self.bot_position)
        )

    @staticmethod
    def _empty_stage_five_rewards() -> dict[str, float | bool]:
        return {
            "retreat_progress_reward": 0.0,
            "cover_progress_reward": 0.0,
            "safe_eat_reward": 0.0,
            "successful_eat_reward": 0.0,
            "recovered_health": 0.0,
            "health_recovery_reward": 0.0,
            "recovery_completion_reward": 0.0,
            "eating_close_penalty": 0.0,
            "unsafe_eat_penalty": 0.0,
            "no_food_penalty": 0.0,
            "unnecessary_eat_penalty": 0.0,
            "ate_successfully": False,
            "reengage_reward": 0.0,
            "premature_reengage_penalty": 0.0,
            "survival_reward": 0.0,
            "survived_timeout_reward": 0.0,
            "death_penalty": 0.0,
            "defeated_after_recovery": False,
            "premature_defeat_reward_removed": 0.0,
            "recovery_victory_reward": 0.0,
            "action_duration_steps": 0.0,
            "success": False,
        }
