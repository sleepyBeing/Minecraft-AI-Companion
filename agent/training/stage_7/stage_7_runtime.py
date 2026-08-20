"""Stage Seven environment creation."""

from __future__ import annotations

import argparse

from agent.stages.stage_7_live import LiveStageSevenPrioritizationEnv
from agent.stages.stage_7_prioritization import StageSevenPrioritizationEnv
from agent.training.stage_7.stage_7_preflight import run_live_stage_seven_preflight


def create_environment(args: argparse.Namespace) -> tuple[StageSevenPrioritizationEnv, str]:
    if args.simulation:
        return StageSevenPrioritizationEnv(), "simulation"
    environment = LiveStageSevenPrioritizationEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    try:
        run_live_stage_seven_preflight(environment)
    except Exception:
        environment.close()
        raise
    return environment, "minecraft"
