"""Mineflayer-backed Stage Four environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
)
from agent.stages.stage_4_cover import CoverPlan
from agent.stages.stage_4_retreat import StageFourRetreatEnv


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
        self.server_safe_position = np.array([5.75, 3.5], dtype=np.float32)
        self.server_protected_position = self.server_safe_position.copy()
        self.server_active_cover = 0
        self.server_cover_occluded = False

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        health_before_action = self.bot_health
        in_cover_before = self._is_in_cover()
        safe_position_before = self._nearest_safe_position()
        safe_distance_before = float(
            np.linalg.norm(safe_position_before - self.bot_position)
        )
        observation, reward, terminated, truncated, info = (
            LiveStageTwoStationaryCombatEnv.step(self, action)
        )
        del observation
        self._update_cover_confirmation()

        damage_taken = max(0.0, health_before_action - self.bot_health)
        damage_taken_penalty = damage_taken * self.DAMAGE_TAKEN_PENALTY_SCALE
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
            safe_position_before=safe_position_before,
            safe_distance_before=safe_distance_before,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
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
            self.server_safe_position = safe_position
            self.server_protected_position = protected
            self.server_active_cover = active_cover
            self.server_cover_occluded = bool(state["coverOccluded"])
            if bool(state["survivalMode"]) != self.survival_mode:
                raise ValueError(
                    "Mineflayer survival mode does not match the episode mode"
                )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid Stage Four state from Mineflayer: {state!r}"
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
