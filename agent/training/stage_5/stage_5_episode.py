"""Episode accounting and CSV logging for Stage Five."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agent.stages.stage_5_recovery import StageFiveAction


METRIC_NAMES = (
    "damage_dealt",
    "damage_taken",
    "bot_defeats",
    "attack_actions",
    "invalid_attacks",
    "retreat_actions",
    "safe_area_actions",
    "eat_actions",
    "reengage_actions",
    "successful_eats",
    "close_eat_attempts",
    "unsafe_eat_attempts",
    "recovered_health",
    "cover_steps",
    "recovered_steps",
    "safe_eat_reward",
    "successful_eat_reward",
    "health_recovery_reward",
    "recovery_completion_reward",
    "reengage_reward",
    "survival_reward",
    "timeout_reward",
    "death_penalty",
    "recovery_victory_reward",
)


class StageFiveEpisodeState:
    def __init__(self) -> None:
        self.episode_number = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        self.episode_return = 0.0
        self.decision_count = 0
        self.game_steps = 0
        self.metrics = {name: 0.0 for name in METRIC_NAMES}

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        self.episode_return += reward
        self.decision_count += 1
        self.game_steps += int(info["action_duration_steps"])
        values = transition_metrics(action, info)
        for name in METRIC_NAMES:
            self.metrics[name] += values[name]


def transition_metrics(
    action: int,
    info: dict[str, Any],
) -> dict[str, float]:
    selected = StageFiveAction(action)
    return {
        "damage_dealt": float(info["damage_dealt"]),
        "damage_taken": float(info["damage_taken"]),
        "bot_defeats": float(info["bot_defeated"]),
        "attack_actions": float(selected == StageFiveAction.ATTACK),
        "invalid_attacks": float(info["invalid_attack"]),
        "retreat_actions": float(selected == StageFiveAction.RETREAT),
        "safe_area_actions": float(
            selected == StageFiveAction.MOVE_TO_SAFE_AREA
        ),
        "eat_actions": float(selected == StageFiveAction.EAT),
        "reengage_actions": float(selected == StageFiveAction.REENGAGE),
        "successful_eats": float(info["ate_successfully"]),
        "close_eat_attempts": float(info["eating_close_penalty"] > 0),
        "unsafe_eat_attempts": float(info["unsafe_eat_penalty"] > 0),
        "recovered_health": float(info["recovered_health"]),
        "cover_steps": float(info["in_cover"]),
        "recovered_steps": float(info["has_recovered"]),
        "safe_eat_reward": float(info["safe_eat_reward"]),
        "successful_eat_reward": float(info["successful_eat_reward"]),
        "health_recovery_reward": float(info["health_recovery_reward"]),
        "recovery_completion_reward": float(
            info["recovery_completion_reward"]
        ),
        "reengage_reward": float(info["reengage_reward"]),
        "survival_reward": float(info["survival_reward"]),
        "timeout_reward": float(info["survived_timeout_reward"]),
        "death_penalty": float(info["death_penalty"]),
        "recovery_victory_reward": float(info["recovery_victory_reward"]),
    }


def finish_episode(
    *,
    tf: Any,
    info: dict[str, Any],
    state: StageFiveEpisodeState,
    logger: "StageFiveEpisodeCsvLogger",
    summary_writer: Any,
) -> None:
    state.episode_number += 1
    if bool(info["bot_defeated"]):
        outcome = "death"
    elif bool(info["defeated_after_recovery"]):
        outcome = "recovery_victory"
    elif not bool(info["target_alive"]):
        outcome = "premature_victory"
    elif float(info["survived_timeout_reward"]) > 0:
        outcome = "survived_timeout"
    else:
        outcome = "timeout"

    logger.write(
        episode=state.episode_number,
        outcome=outcome,
        info=info,
        state=state,
    )
    with summary_writer.as_default():
        summary_values = {
            "return": state.episode_return,
            "decisions": state.decision_count,
            "game_steps": state.game_steps,
            "success": float(info["success"]),
            "starting_health": 15.0,
            "final_health": float(info["bot_health"]),
            "food_remaining": float(info["food_count"]),
            "has_recovered": float(info["has_recovered"]),
            **state.metrics,
        }
        for name, value in summary_values.items():
            tf.summary.scalar(
                f"episode/{name}", value, step=state.episode_number
            )


class StageFiveEpisodeCsvLogger:
    COLUMNS = (
        "episode",
        "return",
        "decisions",
        "game_steps",
        "success",
        "outcome",
        "final_health",
        "food_remaining",
        "has_recovered",
        "final_distance",
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
        state: StageFiveEpisodeState,
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
                info["food_count"],
                int(info["has_recovered"]),
                info["distance_to_target"],
                info["target_health"],
                *(state.metrics[name] for name in METRIC_NAMES),
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()
