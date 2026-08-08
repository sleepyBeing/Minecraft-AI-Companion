"""Environment creation for simulated and live Stage Four training."""

from __future__ import annotations

import argparse

from agent.stages.stage_4_live import LiveStageFourRetreatEnv
from agent.stages.stage_4_retreat import StageFourRetreatEnv
from agent.training.stage_4.stage_4_preflight import run_live_stage_four_preflight


def create_environment(
    args: argparse.Namespace,
) -> tuple[StageFourRetreatEnv, str]:
    if args.simulation:
        return StageFourRetreatEnv(), "simulation"
    environment = LiveStageFourRetreatEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    environment.bridge.connect()
    run_live_stage_four_preflight(environment)
    return environment, "minecraft"
