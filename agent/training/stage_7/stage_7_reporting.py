"""Configuration, TensorBoard, and terminal reporting for Stage Seven."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.training.stage_6.stage_6_reporting import (
    write_rollout_summaries as write_stage_six_summaries,
)


def write_configuration(*, args: argparse.Namespace, source: str,
                        observation_size: int, action_count: int,
                        model_directory: Path) -> None:
    configuration = vars(args).copy()
    for name in ("resume", "stage6_checkpoint"):
        configuration[name] = str(configuration[name]) if configuration[name] else None
    configuration.update({
        "stage": 7,
        "task": "two_threat_target_prioritization",
        "source": source,
        "algorithm": "PPO",
        "framework": "TensorFlow",
        "observation_size": observation_size,
        "action_count": action_count,
        "duration_aware_discounting": True,
        "architecture": "separate_actor_and_critic",
        "actor_initialization": (
            "stage_7_resume" if args.resume else
            "expanded_stage_6_actor" if args.stage6_checkpoint else "fresh"
        ),
        "critic_initialization": "stage_7_resume" if args.resume else "fresh",
    })
    (model_directory / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )


def write_rollout_summaries(*, tf: Any, summary_writer: Any,
                            metrics: dict[str, float],
                            rollout: dict[str, np.ndarray], step: int) -> None:
    write_stage_six_summaries(
        tf=tf, summary_writer=summary_writer, metrics=metrics,
        rollout=rollout, step=step,
    )
    selections = rollout["select_enemy_1_actions"] + rollout["select_enemy_2_actions"]
    with summary_writer.as_default():
        values = {
            "prioritization/selector_actions": np.sum(selections),
            "prioritization/select_enemy_1_actions": np.sum(rollout["select_enemy_1_actions"]),
            "prioritization/select_enemy_2_actions": np.sum(rollout["select_enemy_2_actions"]),
            "prioritization/correct_target_fraction": np.mean(
                rollout["priority_target_selected_steps"]
            ),
            "prioritization/enemy_neutralizations": np.sum(
                rollout["enemy_neutralizations"]
            ),
            "prioritization/highest_priority_neutralizations": np.sum(
                rollout["priority_neutralizations"]
            ),
            "prioritization/wrong_priority_neutralizations": np.sum(
                rollout["wrong_priority_neutralizations"]
            ),
            "prioritization/neutralization_rewards": np.sum(
                rollout["priority_neutralization_reward"]
            ),
            "prioritization/target_selection_rewards": np.sum(
                rollout["target_selection_reward"]
            ),
            "prioritization/correct_target_switches": np.sum(
                rollout["correct_target_switches"]
            ),
            "prioritization/wrong_target_switches": np.sum(
                rollout["wrong_target_switches"]
            ),
            "prioritization/redundant_target_selections": np.sum(
                rollout["redundant_target_selections"]
            ),
            "prioritization/enemies_attacking_npc": np.mean(
                rollout["enemy_attacking_npc_steps"]
            ),
            "outcome/all_enemies_defeated": np.sum(rollout["all_enemies_defeated"]),
        }
        for name, value in values.items():
            tf.summary.scalar(name, value, step=step)
    summary_writer.flush()


def print_rollout(step: int, timesteps: int, rollout: dict[str, np.ndarray],
                  metrics: dict[str, float]) -> None:
    print(
        f"step={step:,}/{timesteps:,} "
        f"reward={np.mean(rollout['rewards']):+.3f} "
        f"npc_damage={np.sum(rollout['npc_damage_taken']):.1f} "
        f"npc_deaths={int(np.sum(rollout['npc_defeats']))} "
        f"victories={int(np.sum(rollout['all_enemies_defeated']))} "
        f"neutralized={int(np.sum(rollout['enemy_neutralizations']))} "
        f"priority_kills={int(np.sum(rollout['priority_neutralizations']))} "
        f"wrong_priority={int(np.sum(rollout['wrong_priority_neutralizations']))} "
        f"correct_target={np.mean(rollout['priority_target_selected_steps']):.2f} "
        f"select1={int(np.sum(rollout['select_enemy_1_actions']))} "
        f"select2={int(np.sum(rollout['select_enemy_2_actions']))} "
        f"correct_switch={int(np.sum(rollout['correct_target_switches']))} "
        f"wrong_switch={int(np.sum(rollout['wrong_target_switches']))} "
        f"reselect={int(np.sum(rollout['redundant_target_selections']))} "
        f"actor_epochs={int(metrics['actor_epochs_completed'])} "
        f"kl_stop={int(metrics['actor_early_stopped'])} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )
