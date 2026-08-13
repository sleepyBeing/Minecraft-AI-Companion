"""Mineflayer-backed Stage Five recovery environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
)
from agent.stages.stage_4_cover import CoverPlan
from agent.stages.stage_5_recovery import (
    StageFiveAction,
    StageFiveRecoveryEnv,
)


class LiveStageFiveRecoveryEnv(
    LiveStageTwoStationaryCombatEnv,
    StageFiveRecoveryEnv,
):
    """Stage Five rewards backed by authoritative Minecraft state."""

    BRIDGE_STAGE = "stage5"
    BRIDGE_STAGE_NAME = "stage-five"

    def __init__(
        self,
        bridge_url: str = "ws://127.0.0.1:8765",
        *,
        bridge_timeout: float = 20.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            bridge_url=bridge_url,
            bridge_timeout=bridge_timeout,
            render_mode=render_mode,
        )
        self.server_safe_position = np.array([5.75, 3.5], dtype=np.float32)
        self.server_protected_position = self.server_safe_position.copy()
        self.server_active_cover = 0
        self.server_cover_occluded = False
        self.server_eat_result: dict[str, Any] = {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        selected = StageFiveAction(action)
        health_before = self.bot_health
        food_count_before = self.food_count
        recovered_before = self.has_recovered
        target_distance_before = self._distance()
        safe_distance_before = self._safe_distance()
        in_cover_before = self.confirmed_in_cover
        was_in_attack_range = self._is_in_attack_range()
        elapsed_before = self.elapsed_seconds

        observation, reward, terminated, truncated, info = (
            LiveStageTwoStationaryCombatEnv.step(self, action)
        )
        del observation
        self._update_cover_confirmation()
        duration_steps = max(
            1,
            round((self.elapsed_seconds - elapsed_before) / self.STEP_SECONDS),
        )
        damage_taken = max(0.0, health_before - self.bot_health)
        bot_defeated = self.bot_defeated or self.bot_health <= 0
        terminated = terminated or bot_defeated
        truncated = truncated and not terminated
        info.update(
            {
                "damage_taken": damage_taken,
                "damage_taken_penalty": 0.0,
                "bot_defeated": bot_defeated,
                "ate_successfully": bool(
                    self.server_eat_result.get("consumed", False)
                ),
                "eat_interrupted": bool(
                    self.server_eat_result.get("interrupted", False)
                ),
                "eat_no_food": bool(
                    self.server_eat_result.get("noFood", False)
                ),
                "eat_result": dict(self.server_eat_result),
            }
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
            source="minecraft",
        )

    def _apply_live_state(self, state: dict[str, Any]) -> None:
        super()._apply_live_state(state)
        try:
            safe_position = self._position_from_state(state, "safePosition")
            protected = self._position_from_state(state, "protectedPosition")
            active_cover = int(state["activeCover"])
            if active_cover not in (0, 1):
                raise ValueError("activeCover must be zero or one")
            food_count = int(state["foodCount"])
            if not 0 <= food_count <= self.MAX_FOOD_COUNT:
                raise ValueError("foodCount is outside the Stage Five range")
            eat_result = state["eatResult"]
            if not isinstance(eat_result, dict):
                raise ValueError("eatResult must be an object")

            self.server_safe_position = safe_position
            self.server_protected_position = protected
            self.server_active_cover = active_cover
            self.server_cover_occluded = bool(state["coverOccluded"])
            self.food_count = food_count
            self.has_eaten = bool(state["hasEaten"])
            self.has_recovered = bool(state["hasRecovered"])
            self.server_eat_result = dict(eat_result)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid Stage Five state from Mineflayer: {state!r}"
            ) from error

    @staticmethod
    def _position_from_state(
        state: dict[str, Any], key: str
    ) -> np.ndarray:
        position = np.asarray(state[key], dtype=np.float32)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{key} must contain two finite coordinates")
        return position

    def _nearest_safe_position(self) -> np.ndarray:
        return self.server_safe_position.copy()

    def _protected_safe_position(self) -> np.ndarray:
        return self.server_protected_position.copy()

    def _cover_plan(self) -> CoverPlan:
        return CoverPlan(
            cover_index=self.server_active_cover,
            navigation_position=self.server_safe_position.copy(),
            protected_position=self.server_protected_position.copy(),
        )

    def _raw_cover_occlusion(self) -> bool:
        return self.server_cover_occluded
