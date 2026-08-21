"""Episode accounting and CSV logging for Stage Seven."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from agent.stages.stage_5_recovery import StageFiveAction
from agent.stages.stage_7_threats import StageSevenAction
from agent.training.stage_5.stage_5_episode import transition_metrics as stage_five_metrics
from agent.training.stage_6.stage_6_episode import (
    METRIC_NAMES as STAGE_SIX_METRICS,
    StageSixEpisodeState,
    protection_transition_metrics,
)


PRIORITY_METRICS = (
    "enemy_neutralizations",
    "priority_neutralizations",
    "wrong_priority_neutralizations",
    "priority_neutralization_reward",
    "select_enemy_1_actions",
    "select_enemy_2_actions",
    "priority_target_selected_steps",
    "enemy_attacking_npc_steps",
    "enemy_attacking_bot_steps",
    "all_enemies_defeated",
    "target_selection_reward",
    "correct_target_switches",
    "wrong_target_switches",
    "redundant_target_selections",
)
METRIC_NAMES = (*STAGE_SIX_METRICS, *PRIORITY_METRICS)


def inherited_transition_metrics(action: int, info: dict[str, Any]) -> dict[str, float]:
    inherited_action = action if action < 12 else int(StageFiveAction.WAIT)
    return {
        **stage_five_metrics(inherited_action, info),
        **protection_transition_metrics(info),
    }


def priority_transition_metrics(action: int, info: dict[str, Any]) -> dict[str, float]:
    neutralized = int(info["neutralized_enemy"])
    selected = int(info["selected_enemy"])
    highest = int(info["highest_priority_enemy"])
    return {
        "enemy_neutralizations": float(neutralized >= 0),
        "priority_neutralizations": float(info["neutralized_highest_priority"]),
        "wrong_priority_neutralizations": float(neutralized >= 0 and neutralized != highest),
        "priority_neutralization_reward": float(info["priority_neutralization_reward"]),
        "select_enemy_1_actions": float(action == int(StageSevenAction.SELECT_ENEMY_1)),
        "select_enemy_2_actions": float(action == int(StageSevenAction.SELECT_ENEMY_2)),
        "priority_target_selected_steps": float(selected == highest),
        "enemy_attacking_npc_steps": float(np.sum(info["enemy_attacking_npc"])),
        "enemy_attacking_bot_steps": float(np.sum(info["enemy_attacking_bot"])),
        "all_enemies_defeated": float(info["all_enemies_defeated"]),
        "target_selection_reward": float(info["target_selection_reward"]),
        "correct_target_switches": float(info["correct_target_switch"]),
        "wrong_target_switches": float(info["wrong_target_switch"]),
        "redundant_target_selections": float(info["redundant_target_selection"]),
    }


class StageSevenEpisodeState(StageSixEpisodeState):
    def reset_episode(self) -> None:
        super().reset_episode()
        self.metrics.update({name: 0.0 for name in PRIORITY_METRICS})

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        inherited_action = action if action < 12 else int(StageFiveAction.WAIT)
        super().update(inherited_action, reward, info)
        for name, value in priority_transition_metrics(action, info).items():
            self.metrics[name] += value


def finish_episode(*, tf: Any, info: dict[str, Any], state: StageSevenEpisodeState,
                   logger: "StageSevenEpisodeCsvLogger", summary_writer: Any) -> None:
    state.episode_number += 1
    if bool(info["npc_defeated"]):
        outcome = "npc_death"
    elif bool(info["bot_defeated"]):
        outcome = "bot_death"
    elif bool(info["all_enemies_defeated"]):
        outcome = "prioritized_victory"
    elif bool(info["npc_survived_episode"]):
        outcome = "npc_survived_timeout"
    else:
        outcome = "timeout"
    logger.write(episode=state.episode_number, outcome=outcome, info=info, state=state)
    with summary_writer.as_default():
        values = {
            "return": state.episode_return,
            "decisions": state.decision_count,
            "game_steps": state.game_steps,
            "success": float(info["success"]),
            "final_bot_health": float(info["bot_health"]),
            "final_npc_health": float(info["npc_health"]),
            "food_remaining": float(info["food_count"]),
            **state.metrics,
        }
        for name, value in values.items():
            tf.summary.scalar(f"episode/{name}", value, step=state.episode_number)


class StageSevenEpisodeCsvLogger:
    COLUMNS = (
        "episode", "return", "decisions", "game_steps", "success", "outcome",
        "enemy_1_type", "enemy_2_type", "enemy_1_health", "enemy_2_health",
        "final_bot_health", "final_npc_health", "food_remaining", *METRIC_NAMES,
    )

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def write(self, *, episode: int, outcome: str, info: dict[str, Any],
              state: StageSevenEpisodeState) -> None:
        self._writer.writerow([
            episode, state.episode_return, state.decision_count, state.game_steps,
            int(info["success"]), outcome, *info["enemy_types"], *info["enemy_health"],
            info["bot_health"], info["npc_health"], info["food_count"],
            *(state.metrics[name] for name in METRIC_NAMES),
        ])
        self._file.flush()

    def close(self) -> None:
        self._file.close()
