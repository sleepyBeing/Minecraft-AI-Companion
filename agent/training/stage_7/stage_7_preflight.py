"""Live Minecraft contract checks required before Stage Seven training."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_7_live import LiveStageSevenPrioritizationEnv
from agent.stages.stage_7_threats import StageSevenAction


SCENARIO = {
    "enemy_types": ["zombie", "skeleton"],
    "bot_position": [2.0, 7.0],
    "enemy_positions": [[7.0, 7.0], [12.5, 7.0]],
    "npc_position": [9.5, 7.0],
}


def _fail(environment: LiveStageSevenPrioritizationEnv, message: str) -> None:
    environment.close()
    raise RuntimeError(f"Stage-seven preflight failed: {message}")


def run_live_stage_seven_preflight(environment: LiveStageSevenPrioritizationEnv) -> None:
    observation, info = environment.reset(
        options={**SCENARIO, "rebuild_arena": True, "freeze_target": True}
    )
    if observation.shape != (60,) or environment.action_space.n != 14:
        _fail(environment, "the live observation/action contract is not 60/14.")
    if not np.isclose(float(info["bot_health"]), 20.0, atol=0.75):
        _fail(environment, "the bot did not start at full health.")
    if not np.isclose(float(info["npc_health"]), 20.0, atol=0.75):
        _fail(environment, "the NPC did not start at full health.")
    if int(info.get("food_count", 0)) < 64 or not bool(info.get("has_iron_sword")):
        _fail(environment, "the bot did not receive an iron sword and 64 steaks.")
    if tuple(info["enemy_types"]) != ("zombie", "skeleton"):
        _fail(environment, "the two requested enemy types were not created.")
    if not bool(np.all(info["enemy_alive"])):
        _fail(environment, "one or both enemies were missing after reset.")

    _, _, terminated, truncated, info = environment.step(
        int(StageSevenAction.SELECT_ENEMY_2)
    )
    if terminated or truncated or int(info["selected_enemy"]) != 1:
        _fail(environment, "SELECT_ENEMY_2 did not switch targets safely.")
    _, _, terminated, truncated, info = environment.step(
        int(StageSevenAction.SELECT_ENEMY_1)
    )
    if terminated or truncated or int(info["selected_enemy"]) != 0:
        _fail(environment, "SELECT_ENEMY_1 did not switch targets safely.")

    _, info = environment.reset(options={**SCENARIO, "freeze_target": False})
    npc_start = np.asarray(info["npc_position"], dtype=np.float32)
    enemy_start = np.asarray(info["enemy_positions"], dtype=np.float32)
    moved = False
    for _ in range(10):
        _, _, terminated, truncated, info = environment.step(int(StageSevenAction.WAIT))
        if np.linalg.norm(np.asarray(info["npc_position"]) - npc_start) > 0.25:
            _fail(environment, "the protected NPC did not remain stationary.")
        moved |= bool(np.any(np.linalg.norm(
            np.asarray(info["enemy_positions"]) - enemy_start, axis=1
        ) > 0.05))
        if terminated or truncated:
            break
    if not moved:
        _fail(environment, "neither enemy moved toward the interaction.")
    print(
        "Stage-seven preflight passed: equipment, two target selectors, "
        "stationary NPC, and moving enemies are working."
    )
