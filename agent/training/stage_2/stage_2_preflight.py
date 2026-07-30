"""Preflight testing for Stage Two combat."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
    StationaryCombatAction,
)


def run_live_combat_preflight(
    environment: LiveStageTwoStationaryCombatEnv,
) -> None:
    """Prove that a controlled in-range sword attack damages the live zombie."""

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

    for _ in range(30):
        _, _, terminated, truncated, info = environment.step(
            int(StationaryCombatAction.FORWARD)
        )
        if terminated or truncated:
            break
        closest_distance = min(
            closest_distance, float(info["distance_to_target"])
        )
        if float(info["distance_to_target"]) <= 2.4:
            break

    if closest_distance >= starting_distance - 0.25:
        environment.close()
        raise RuntimeError(
            "Stage-two combat preflight failed: the bot could not approach "
            "the stationary zombie. Check Mineflayer movement and arena setup."
        )
    if float(info["distance_to_target"]) > environment.ATTACK_RANGE:
        environment.close()
        raise RuntimeError(
            "Stage-two combat preflight failed: the bot could not enter "
            "server-safe attack range."
        )

    for _ in range(10):
        if bool(info["attack_ready"]):
            break
        _, _, _, _, info = environment.step(int(StationaryCombatAction.WAIT))

    _, _, _, _, attack_info = environment.step(
        int(StationaryCombatAction.ATTACK)
    )
    if (
        not bool(attack_info["valid_attack_attempt"])
        or float(attack_info["damage_dealt"]) <= 0
    ):
        environment.close()
        raise RuntimeError(
            "Stage-two combat preflight failed: an in-range iron-sword attack "
            "did not reduce the zombie's server-reported health. PPO was not "
            "started because it would receive no combat reward."
        )

    print(
        "Live combat preflight passed: "
        f"damage={float(attack_info['damage_dealt']):.1f}, "
        f"distance={float(attack_info['distance_to_target']):.2f}"
    )
