"""Stage Six environment creation."""

from __future__ import annotations

import argparse

from agent.stages.stage_6_live import LiveStageSixProtectionEnv
from agent.stages.stage_6_protection import StageSixProtectionEnv
from agent.training.stage_6.stage_6_preflight import run_live_stage_six_preflight


def create_environment(
    args: argparse.Namespace,
) -> tuple[StageSixProtectionEnv, str]:
    if args.simulation:
        return StageSixProtectionEnv(), "simulation"

    environment = LiveStageSixProtectionEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    try:
        run_live_stage_six_preflight(environment)
    except Exception:
        environment.close()
        raise
    return environment, "minecraft"
