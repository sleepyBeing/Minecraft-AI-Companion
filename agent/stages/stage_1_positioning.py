"""Stage one: live Minecraft basic-positioning environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.environment import BasicPositioningEnv
from agent.minecraft_bridge import MinecraftWebSocketBridge


class LiveBasicPositioningEnv(BasicPositioningEnv):
    """Run the stage-one task using the real Mineflayer bot.

    Start the Minecraft server and TypeScript bot before constructing this
    environment. The bot hosts its WebSocket bridge on localhost port 8765.
    """

    def __init__(
        self,
        bridge_url: str = "ws://127.0.0.1:8765",
        *,
        bridge_timeout: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.bridge = MinecraftWebSocketBridge(bridge_url, timeout=bridge_timeout)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = options or {}
        # The simulated parent reset supplies deterministic, validated starting
        # positions. Mineflayer then reproduces those positions in the arena.
        super().reset(seed=seed, options=options)
        response = self.bridge.request(
            "stage1.reset",
            botPosition=self.bot_position.tolist(),
            targetPosition=self.target_position.tolist(),
            botYaw=self.bot_yaw,
            rebuildArena=bool(options.get("rebuild_arena", False)),
        )
        self._apply_live_state(response["state"])
        self.previous_distance = self._distance()
        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer from 0 to 6")

        old_position = self.bot_position.copy()
        old_distance = self.previous_distance
        response = self.bridge.request("stage1.step", action=int(action))
        self._apply_live_state(response["state"])

        distance = self._distance()
        distance_change = old_distance - distance
        reward = self.PROGRESS_REWARD_SCALE * distance_change - self.TIME_PENALTY
        terminated = distance <= self.ATTACK_RANGE
        if terminated and old_distance > self.ATTACK_RANGE:
            reward += self.ATTACK_RANGE_REWARD

        truncated = self.elapsed_seconds >= self.EPISODE_SECONDS and not terminated
        self.previous_distance = distance
        info = self._get_info()
        info.update(
            {
                "distance_change": distance_change,
                "position_changed": bool(
                    np.linalg.norm(self.bot_position - old_position) > 1e-6
                ),
                "success": terminated,
                "source": "minecraft",
            }
        )
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self.bridge.close()

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
            self.has_equipment = bool(state["hasEquipment"])
            self.elapsed_seconds = float(state["elapsedSeconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid stage-one state from Mineflayer: {state!r}") from error
