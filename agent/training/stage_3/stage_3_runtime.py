"""Environment creation for simulated and live Stage Three training."""

from __future__ import annotations

import argparse

from agent.stages.stage_3_moving_combat import (
    LiveStageThreeMovingCombatEnv,
    StageThreeMovingCombatEnv,
)
from agent.training.stage_3.stage_3_preflight import (
    run_live_moving_combat_preflight,
)


def create_environment(
    args: argparse.Namespace,
) -> tuple[StageThreeMovingCombatEnv, str]:
    if args.simulation:
        return StageThreeMovingCombatEnv(), "simulation"
    environment = LiveStageThreeMovingCombatEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    environment.bridge.connect()
    run_live_moving_combat_preflight(environment)
    return environment, "minecraft"
