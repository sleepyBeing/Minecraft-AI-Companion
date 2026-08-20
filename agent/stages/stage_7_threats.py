"""Enemy definitions and hidden threat-ranking rules for Stage Seven."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


THREAT_TYPES = ("zombie", "skeleton", "enderman", "spider", "witch")


class StageSevenAction(IntEnum):
    WAIT = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    ATTACK = 7
    RETREAT = 8
    MOVE_TO_SAFE_AREA = 9
    EAT = 10
    REENGAGE = 11
    SELECT_ENEMY_1 = 12
    SELECT_ENEMY_2 = 13


@dataclass(frozen=True)
class ThreatSpec:
    max_health: float
    move_speed: float
    attack_range: float
    attack_damage: float
    cooldown_seconds: float
    ranged: bool = False


THREAT_SPECS = {
    "zombie": ThreatSpec(20.0, 2.3, 1.8, 3.0, 1.0),
    "skeleton": ThreatSpec(20.0, 2.0, 7.0, 2.0, 1.25, True),
    "enderman": ThreatSpec(40.0, 3.0, 1.9, 4.0, 1.0),
    "spider": ThreatSpec(16.0, 3.2, 1.8, 2.0, 0.8),
    "witch": ThreatSpec(26.0, 1.7, 6.0, 3.0, 1.5, True),
}


def threat_priority(
    *,
    bot_distance: float,
    npc_distance: float,
    health: float,
    max_health: float,
    ranged: bool,
    attacking_npc: bool,
    attacking_bot: bool,
    arena_diagonal: float,
) -> float:
    """Return the hidden teacher score; NPC danger dominates every factor."""
    npc_proximity = 1.0 - min(1.0, npc_distance / arena_diagonal)
    bot_proximity = 1.0 - min(1.0, bot_distance / arena_diagonal)
    vulnerable = 1.0 - min(1.0, health / max_health)
    return (
        10.0 * float(attacking_npc)
        + 4.0 * npc_proximity
        + 2.0 * float(ranged)
        + 2.0 * float(attacking_bot)
        + 1.5 * bot_proximity
        + vulnerable
    )
