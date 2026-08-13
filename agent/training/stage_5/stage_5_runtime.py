"""Stage Five environment creation."""

from __future__ import annotations

from agent.stages.stage_5_recovery import StageFiveRecoveryEnv


def create_environment() -> tuple[StageFiveRecoveryEnv, str]:
    return StageFiveRecoveryEnv(), "simulation"
