"""Live arena checks before Stage Four training begins."""

from __future__ import annotations

import numpy as np

from agent.stages.stage_2_stationary_combat import StationaryCombatAction
from agent.stages.stage_4_retreat import LiveStageFourRetreatEnv


BASE_OPTIONS = {
    "bot_position": [4.5, 7.5],
    "target_position": [10.5, 7.5],
    "bot_yaw": -np.pi / 2.0,
}


def run_live_stage_four_preflight(
    environment: LiveStageFourRetreatEnv,
) -> None:
    """Verify low health, extended actions, cover state, and live combat"""

    observation, info = environment.reset(
        seed=0,
        options={**BASE_OPTIONS, "bot_health": 5},
    )
    if observation.shape != (18,) or environment.action_space.n != 10:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: expected 18 observations and 10 actions."
        )
    if abs(float(info["bot_health"]) - 5.0) > 0.75:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: the five-health scenario was not applied."
        )

    distance_before_retreat = float(info["distance_to_target"])
    furthest_retreat_distance = distance_before_retreat
    retreat_info = info
    for _ in range(5):
        _, _, terminated, truncated, retreat_info = environment.step(
            int(StationaryCombatAction.RETREAT)
        )
        if terminated or truncated:
            environment.close()
            raise RuntimeError("Stage-four preflight failed during retreat actions.")
        furthest_retreat_distance = max(
            furthest_retreat_distance,
            float(retreat_info["distance_to_target"]),
        )
    if furthest_retreat_distance < distance_before_retreat + 0.25:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: sustained RETREAT did not create distance."
        )

    _, info = environment.reset(
        seed=1,
        options={**BASE_OPTIONS, "bot_health": 5},
    )
    safe_distance_before = float(info["distance_to_safe_position"])
    _, _, terminated, truncated, safe_info = environment.step(
        int(StationaryCombatAction.MOVE_TO_SAFE_AREA)
    )
    if terminated or truncated:
        environment.close()
        raise RuntimeError("Stage-four preflight failed during safe-area movement.")
    if float(safe_info["distance_to_safe_position"]) >= safe_distance_before:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: MOVE_TO_SAFE_AREA did not approach cover."
        )
    if float(safe_info["cover_progress_reward"]) <= 0:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: approaching cover earned no progress reward."
        )

    _, info = environment.reset(
        seed=2,
        options={**BASE_OPTIONS, "bot_health": 20},
    )
    starting_distance = float(info["distance_to_target"])
    total_damage_taken = 0.0
    for _ in range(45):
        _, _, terminated, truncated, info = environment.step(
            int(StationaryCombatAction.WAIT)
        )
        total_damage_taken += float(info["damage_taken"])
        if terminated or truncated or total_damage_taken > 0:
            break

    if float(info["distance_to_target"]) >= starting_distance - 0.5:
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: the zombie did not pursue the bot."
        )
    if total_damage_taken <= 0 or bool(info.get("bot_defeated", False)):
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: incoming zombie damage was not verified."
        )

    for _ in range(10):
        if bool(info["in_attack_range"]) and bool(info["attack_ready"]):
            break
        _, _, terminated, truncated, info = environment.step(
            int(StationaryCombatAction.WAIT)
        )
        if terminated or truncated:
            break
    if not bool(info["in_attack_range"]):
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: the zombie never entered the "
            "bridge-valid attack range."
        )

    _, _, _, _, attack_info = environment.step(
        int(StationaryCombatAction.ATTACK)
    )
    if (
        not bool(attack_info["valid_attack_attempt"])
        or float(attack_info["damage_dealt"]) <= 0
    ):
        environment.close()
        raise RuntimeError(
            "Stage-four preflight failed: an in-range sword attack caused no damage."
        )

    print(
        "Live Stage Four preflight passed: "
        f"retreat_gain={furthest_retreat_distance - distance_before_retreat:.2f}, "
        f"safe_progress={safe_distance_before - float(safe_info['distance_to_safe_position']):.2f}, "
        f"damage_taken={total_damage_taken:.1f}, "
        f"damage_dealt={float(attack_info['damage_dealt']):.1f}"
    )
