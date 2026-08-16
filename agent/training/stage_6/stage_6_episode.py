"""Episode accounting and CSV logging for Stage Six."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agent.training.stage_5.stage_5_episode import (
    METRIC_NAMES as STAGE_FIVE_METRICS,
    StageFiveEpisodeState,
    transition_metrics as stage_five_transition_metrics,
)


PROTECTION_METRICS = (
    "npc_damage_taken",
    "npc_damage_penalty",
    "npc_defeats",
    "npc_survival_reward",
    "npc_survived_episodes",
    "bot_damage_penalty",
    "stray_penalty",
    "npc_under_attack_steps",
)
METRIC_NAMES = (*STAGE_FIVE_METRICS, *PROTECTION_METRICS)


def protection_transition_metrics(info: dict[str, Any]) -> dict[str, float]:
    return {
        "npc_damage_taken": float(info["npc_damage_taken"]),
        "npc_damage_penalty": float(info["npc_damage_penalty"]),
        "npc_defeats": float(info["npc_defeated"]),
        "npc_survival_reward": float(info["npc_survival_reward"]),
        "npc_survived_episodes": float(info["npc_survived_episode"]),
        "bot_damage_penalty": float(info["bot_damage_penalty"]),
        "stray_penalty": float(info["stray_penalty"]),
        "npc_under_attack_steps": float(info["npc_under_attack"]),
    }


class StageSixEpisodeState(StageFiveEpisodeState):
    def reset_episode(self) -> None:
        super().reset_episode()
        self.metrics.update({name: 0.0 for name in PROTECTION_METRICS})

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        super().update(action, reward, info)
        for name, value in protection_transition_metrics(info).items():
            self.metrics[name] += value


def finish_episode(
    *,
    tf: Any,
    info: dict[str, Any],
    state: StageSixEpisodeState,
    logger: "StageSixEpisodeCsvLogger",
    summary_writer: Any,
) -> None:
    state.episode_number += 1
    if bool(info["npc_defeated"]):
        outcome = "npc_death"
    elif bool(info["bot_defeated"]):
        outcome = "bot_death"
    elif not bool(info["target_alive"]):
        outcome = "protected_victory"
    elif bool(info["npc_survived_episode"]):
        outcome = "npc_survived_timeout"
    else:
        outcome = "timeout"

    logger.write(
        episode=state.episode_number,
        outcome=outcome,
        info=info,
        state=state,
    )
    with summary_writer.as_default():
        values = {
            "return": state.episode_return,
            "decisions": state.decision_count,
            "game_steps": state.game_steps,
            "success": float(info["success"]),
            "final_bot_health": float(info["bot_health"]),
            "final_npc_health": float(info["npc_health"]),
            "food_remaining": float(info["food_count"]),
            "has_recovered": float(info["has_recovered"]),
            **state.metrics,
        }
        for name, value in values.items():
            tf.summary.scalar(
                f"episode/{name}", value, step=state.episode_number
            )


class StageSixEpisodeCsvLogger:
    COLUMNS = (
        "episode",
        "return",
        "decisions",
        "game_steps",
        "success",
        "outcome",
        "final_bot_health",
        "final_npc_health",
        "food_remaining",
        "has_recovered",
        "bot_enemy_distance",
        "bot_npc_distance",
        "enemy_npc_distance",
        "target_health",
        *METRIC_NAMES,
    )

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def write(
        self,
        *,
        episode: int,
        outcome: str,
        info: dict[str, Any],
        state: StageSixEpisodeState,
    ) -> None:
        self._writer.writerow(
            [
                episode,
                state.episode_return,
                state.decision_count,
                state.game_steps,
                int(info["success"]),
                outcome,
                info["bot_health"],
                info["npc_health"],
                info["food_count"],
                int(info["has_recovered"]),
                info["distance_to_target"],
                info["bot_npc_distance"],
                info["enemy_npc_distance"],
                info["target_health"],
                *(state.metrics[name] for name in METRIC_NAMES),
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
