"""Live Minecraft contract checks required before Stage Six training."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_5_recovery import StageFiveAction
from agent.stages.stage_6_live import LiveStageSixProtectionEnv


def _fail(environment: LiveStageSixProtectionEnv, message: str) -> None:
    environment.close()
    raise RuntimeError(f"Stage-six preflight failed: {message}")


def run_live_stage_six_preflight(
    environment: LiveStageSixProtectionEnv,
) -> None:
    observation, info = environment.reset(options={"rebuild_arena": True})
    if observation.shape != (30,) or environment.action_space.n != 12:
        _fail(environment, "the live observation/action contract is not 30/12.")
    if not np.isclose(float(info["bot_health"]), 20.0, atol=0.75):
        _fail(environment, "the bot did not start at full health.")
    if not np.isclose(float(info["npc_health"]), 20.0, atol=0.75):
        _fail(environment, "the NPC did not start at full health.")
    if not bool(info["npc_alive"]) or not bool(info["target_alive"]):
        _fail(environment, "the NPC or zombie was missing after reset.")
    if int(info.get("food_count", 0)) < 64:
        _fail(environment, "the bot did not receive its 64 steaks.")

    npc_start = np.asarray(info["npc_position"], dtype=np.float32)
    zombie_start = np.asarray(info["target_position"], dtype=np.float32)
    zombie_moved = False
    for _ in range(12):
        _, _, terminated, truncated, info = environment.step(
            int(StageFiveAction.WAIT)
        )
        if np.linalg.norm(np.asarray(info["npc_position"]) - npc_start) > 0.25:
            _fail(environment, "the protected NPC did not remain stationary.")
        if np.linalg.norm(np.asarray(info["target_position"]) - zombie_start) > 0.05:
            zombie_moved = True
        if terminated or truncated:
            break
    if not zombie_moved:
        _fail(environment, "the zombie did not move toward the interaction.")

    print(
        "Stage-six preflight passed: observations, equipment, stationary NPC, "
        "and moving zombie are working."
    )
