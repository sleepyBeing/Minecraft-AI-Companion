"""Mineflayer-backed Stage Six NPC protection environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_5_live import LiveStageFiveRecoveryEnv
from agent.stages.stage_6_protection import StageSixProtectionEnv


class LiveStageSixProtectionEnv(
    LiveStageFiveRecoveryEnv,
    StageSixProtectionEnv,
):
    
    BRIDGE_STAGE = "stage6"
    BRIDGE_STAGE_NAME = "stage-six"

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        npc_health_before = self.npc_health
        observation, reward, terminated, truncated, info = (
            LiveStageFiveRecoveryEnv.step(self, action)
        )
        del observation
        npc_damage_taken = max(0.0, npc_health_before - self.npc_health)
        return self._apply_protection_rewards(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            npc_damage_taken=npc_damage_taken,
        )

    def _apply_live_state(self, state: dict[str, Any]) -> None:
        super()._apply_live_state(state)
        try:
            npc_position = self._position_from_state(state, "npcPosition")
            npc_health = float(state["npcHealth"])
            if not 0.0 <= npc_health <= self.NPC_MAX_HEALTH:
                raise ValueError("npcHealth is outside health bounds")
            bot_npc_distance = float(state["botNpcDistance"])
            enemy_npc_distance = float(state["enemyNpcDistance"])
            interaction_distance = float(state["interactionDistance"])
            distances = (
                bot_npc_distance,
                enemy_npc_distance,
                interaction_distance,
            )
            if not all(np.isfinite(value) and value >= 0 for value in distances):
                raise ValueError("Stage Six distances must be nonnegative")

            self.npc_position = npc_position
            self.npc_health = npc_health
            self.npc_alive = bool(state["npcAlive"])
            self.npc_under_attack = bool(state["npcUnderAttack"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid Stage Six state from Mineflayer: {state!r}"
            ) from error
