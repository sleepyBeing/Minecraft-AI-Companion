"""Stage Five environment creation."""

from __future__ import annotations

import argparse

from agent.stages.stage_5_live import LiveStageFiveRecoveryEnv
from agent.stages.stage_5_recovery import StageFiveRecoveryEnv
from agent.training.stage_5.stage_5_preflight import run_live_stage_five_preflight


def create_environment(
    args: argparse.Namespace,
) -> tuple[StageFiveRecoveryEnv, str]:
    if args.simulation:
        return StageFiveRecoveryEnv(), "simulation"

    environment = LiveStageFiveRecoveryEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    try:
        run_live_stage_five_preflight(environment)
    except Exception:
        environment.close()
        raise
    return environment, "minecraft"
