"""Live Minecraft checks required before Stage Five PPO training."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_5_live import LiveStageFiveRecoveryEnv
from agent.stages.stage_5_recovery import StageFiveAction


def _fail(environment: LiveStageFiveRecoveryEnv, message: str) -> None:
    environment.close()
    raise RuntimeError(f"Stage-five preflight failed: {message}")


def run_live_stage_five_preflight(
    environment: LiveStageFiveRecoveryEnv,
) -> None:
    observation, info = environment.reset(
        options={"rebuild_arena": True},
    )
    if observation.shape != (22,) or environment.action_space.n != 12:
        _fail(environment, "the live observation/action contract is not 22/12.")
    if not np.isclose(float(info["bot_health"]), 15.0, atol=0.75):
        _fail(environment, "the bot did not start at 15 health.")
    if int(info.get("food_count", 0)) < 64:
        _fail(environment, "the bot did not receive its 64 steaks.")

    protected_position = np.asarray(
        info.get("protected_position"), dtype=np.float32
    )
    target_position = np.asarray(info.get("target_position"), dtype=np.float32)
    if protected_position.shape != (2,) or target_position.shape != (2,):
        _fail(environment, "the arena did not report a protected position.")

    _, info = environment.reset(
        options={
            "bot_position": (
                float(protected_position[0]),
                float(protected_position[1]),
            ),
            "target_position": (
                float(target_position[0]),
                float(target_position[1]),
            ),
            "rebuild_arena": False,
        }
    )
    for _ in range(3):
        _, _, terminated, truncated, info = environment.step(
            int(StageFiveAction.WAIT)
        )
        if terminated or truncated:
            _fail(environment, "the cover validation episode ended early.")
    if not bool(info.get("safe_to_eat", False)):
        _fail(environment, "the protected position was not classified as safe.")

    food_before = int(info.get("food_count", 0))
    _, eat_reward, terminated, truncated, eat_info = environment.step(
        int(StageFiveAction.EAT)
    )
    if terminated or truncated:
        _fail(environment, "the eating validation episode ended early.")
    if not bool(eat_info.get("ate_successfully", False)):
        result = eat_info.get("eat_result", "unknown")
        _fail(environment, f"the bot did not consume food ({result}).")
    if int(eat_info.get("food_count", food_before)) >= food_before:
        _fail(environment, "eating did not consume an inventory item.")
    if eat_reward <= 0.0:
        _fail(environment, "safe eating did not produce positive reward.")

    recovered_info = eat_info
    for _ in range(50):
        if bool(recovered_info.get("has_recovered", False)):
            break
        _, _, terminated, truncated, recovered_info = environment.step(
            int(StageFiveAction.WAIT)
        )
        if terminated or truncated:
            _fail(environment, "the recovery validation episode ended early.")
    if not bool(recovered_info.get("has_recovered", False)):
        _fail(environment, "real health did not recover after eating.")

    distance_before = float(recovered_info["distance_to_target"])
    _, _, _, _, reengage_info = environment.step(
        int(StageFiveAction.REENGAGE)
    )
    if float(reengage_info["distance_to_target"]) >= distance_before - 0.05:
        _fail(environment, "REENGAGE did not reduce zombie distance.")

    print(
        "Stage-five preflight passed: live health, food consumption, "
        "recovery, cover, and re-engagement are working."
    )
