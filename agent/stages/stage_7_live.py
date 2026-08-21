"""Mineflayer-backed Stage Seven target-prioritization environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.stages.stage_6_live import LiveStageSixProtectionEnv
from agent.stages.stage_7_prioritization import StageSevenPrioritizationEnv
from agent.stages.stage_7_threats import THREAT_SPECS, THREAT_TYPES


class LiveStageSevenPrioritizationEnv(
    LiveStageSixProtectionEnv,
    StageSevenPrioritizationEnv,
):
    BRIDGE_STAGE = "stage7"
    BRIDGE_STAGE_NAME = "stage-seven"

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Invalid Stage Seven action; expected 0 to 13")
        priority_before = self._highest_priority_enemy()
        alive_before = self.enemy_alive.copy()
        selected_before = self.selected_enemy
        if action < 12:
            result = LiveStageSixProtectionEnv.step(self, action)
        else:
            result = self._live_selection_step(action)
        return self._finalize_stage_seven(
            *result,
            action=action,
            priority_before=priority_before,
            alive_before=alive_before,
            selected_before=selected_before,
        )

    def _live_selection_step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        elapsed_before = self.elapsed_seconds
        health_before = self.bot_health
        npc_health_before = self.npc_health
        response = self.bridge.request(f"{self.BRIDGE_STAGE}.step", action=action)
        self._apply_live_state(response["state"])
        duration = max(
            1,
            round((self.elapsed_seconds - elapsed_before) / self.STEP_SECONDS),
        )
        damage_taken = max(0.0, health_before - self.bot_health)
        npc_damage_taken = max(0.0, npc_health_before - self.npc_health)
        bot_defeated = self.bot_defeated or self.bot_health <= 0
        reward = -self.TIME_PENALTY * duration
        survival_reward = 0.0
        death_penalty = self.DEATH_PENALTY if bot_defeated else 0.0
        if not bot_defeated:
            survival_reward = self.SURVIVAL_STEP_REWARD * duration
            reward += survival_reward
        reward -= death_penalty
        timed_out_alive = (
            self.elapsed_seconds >= self.EPISODE_SECONDS
            and not bot_defeated
            and bool(np.any(self.enemy_alive))
        )
        timeout_reward = (
            self.SURVIVED_TIMEOUT_REWARD if timed_out_alive else 0.0
        )
        reward += timeout_reward

        info = self._get_info()
        info.update(self._empty_stage_five_rewards())
        info.update(
            {
                "damage_dealt": 0.0,
                "damage_taken": damage_taken,
                "damage_taken_penalty": 0.0,
                "distance_change": 0.0,
                "approach_reward": 0.0,
                "entered_attack_range": False,
                "attack_selected": False,
                "valid_attack_attempt": False,
                "invalid_attack": False,
                "cooldown_blocked": False,
                "attack_landed": False,
                "confirmed_kill": False,
                "attack_distance": None,
                "server_health_verified": False,
                "movement_settled": True,
                "movement_not_settled": False,
                "target_missing": False,
                "tracking_failure": False,
                "attack_packet_sent": False,
                "bot_defeated": bot_defeated,
                "survival_reward": survival_reward,
                "survived_timeout_reward": timeout_reward,
                "death_penalty": death_penalty,
                "action_duration_steps": duration,
                "source": "minecraft",
            }
        )
        return self._apply_protection_rewards(
            reward=reward,
            terminated=bot_defeated,
            truncated=False,
            info=info,
            npc_damage_taken=npc_damage_taken,
        )

    def _apply_live_state(self, state: dict[str, Any]) -> None:
        super()._apply_live_state(state)
        try:
            enemy_states = state["enemyStates"]
            if not isinstance(enemy_states, list) or len(enemy_states) != 2:
                raise ValueError("enemyStates must contain two entries")
            types: list[str] = []
            positions: list[np.ndarray] = []
            health: list[float] = []
            alive: list[bool] = []
            visible: list[bool] = []
            for enemy in enemy_states:
                enemy_type = str(enemy["type"])
                if enemy_type not in THREAT_TYPES:
                    raise ValueError("unknown Stage Seven enemy type")
                position = self._position_from_state(enemy, "position")
                maximum = THREAT_SPECS[enemy_type].max_health
                enemy_health = float(enemy["health"])
                if not 0.0 <= enemy_health <= maximum:
                    raise ValueError("enemy health is outside type bounds")
                for name in ("botDistance", "npcDistance"):
                    if not np.isfinite(float(enemy[name])) or float(enemy[name]) < 0:
                        raise ValueError(f"{name} must be nonnegative")
                types.append(enemy_type)
                positions.append(position)
                health.append(enemy_health)
                alive.append(bool(enemy["alive"]))
                visible.append(bool(enemy["visible"]))
            if len(set(types)) != 2:
                raise ValueError("Stage Seven enemy types must be different")
            selected = int(state["selectedEnemy"])
            if selected not in (0, 1):
                raise ValueError("selectedEnemy must be zero or one")
            self.enemy_types = types
            self.enemy_positions = np.asarray(positions, dtype=np.float32)
            self.enemy_health = np.asarray(health, dtype=np.float32)
            self.enemy_alive = np.asarray(alive, dtype=np.bool_)
            self.enemy_visible = np.asarray(visible, dtype=np.bool_)
            self.selected_enemy = selected
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid Stage Seven state from Mineflayer: {state!r}"
            ) from error
