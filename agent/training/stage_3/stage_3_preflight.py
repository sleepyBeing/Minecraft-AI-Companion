"""Live preflight checks for Stage Three moving-target combat."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_2_stationary_combat import StationaryCombatAction
from agent.stages.stage_3_moving_combat import LiveStageThreeMovingCombatEnv


def run_live_moving_combat_preflight(
    environment: LiveStageThreeMovingCombatEnv,
) -> None:
    """Verify zombie movement, incoming damage, and outgoing sword damage."""

    _, info = environment.reset(
        seed=0,
        options={
            "bot_position": [4.5, 7.5],
            "target_position": [10.5, 7.5],
            "bot_yaw": -np.pi / 2.0,
        },
    )
    starting_distance = float(info["distance_to_target"])
    closest_distance = starting_distance
    total_damage_taken = 0.0

    for _ in range(45):
        _, _, terminated, truncated, info = environment.step(
            int(StationaryCombatAction.WAIT)
        )
        closest_distance = min(
            closest_distance, float(info["distance_to_target"])
        )
        total_damage_taken += float(info["damage_taken"])
        if terminated or truncated or total_damage_taken > 0:
            break

    if closest_distance >= starting_distance - 0.5:
        environment.close()
        raise RuntimeError(
            "Stage-three preflight failed: the zombie did not move toward "
            "the bot. Ensure Stage 3 uses the updated bridge and normal "
            "zombie AI."
        )
    if total_damage_taken <= 0:
        environment.close()
        raise RuntimeError(
            "Stage-three preflight failed: the moving zombie did not damage "
            "the bot. Check server difficulty and zombie AI."
        )
    if bool(info.get("bot_defeated", False)):
        environment.close()
        raise RuntimeError(
            "Stage-three preflight failed: the bot died before its attack "
            "could be verified."
        )
    if float(info["distance_to_target"]) > environment.ATTACK_RANGE:
        environment.close()
        raise RuntimeError(
            "Stage-three preflight failed: the zombie never entered "
            "server-safe attack range."
        )

    for _ in range(10):
        if bool(info["attack_ready"]):
            break
        _, _, terminated, truncated, info = environment.step(
            int(StationaryCombatAction.WAIT)
        )
        if terminated or truncated:
            break

    _, _, _, _, attack_info = environment.step(
        int(StationaryCombatAction.ATTACK)
    )
    if (
        not bool(attack_info["valid_attack_attempt"])
        or float(attack_info["damage_dealt"]) <= 0
    ):
        environment.close()
        raise RuntimeError(
            "Stage-three preflight failed: an in-range iron-sword attack "
            "did not reduce the moving zombie's server-reported health."
        )

    print(
        "Live Stage Three preflight passed: "
        f"zombie_approach={starting_distance - closest_distance:.2f}, "
        f"damage_taken={total_damage_taken:.1f}, "
        f"damage_dealt={float(attack_info['damage_dealt']):.1f}"
    )
