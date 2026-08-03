"""Configuration, TensorBoard, and terminal reporting for Stage Four."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.training.stage_1.train_stage_1 import write_training_summaries


def write_configuration(
    *,
    args: argparse.Namespace,
    source: str,
    observation_size: int,
    action_count: int,
    model_directory: Path,
) -> None:
    configuration = vars(args).copy()
    for name in ("resume", "stage3_checkpoint"):
        configuration[name] = (
            str(configuration[name]) if configuration[name] else None
        )
    configuration.update(
        {
            "stage": 4,
            "task": "health_aware_combat_retreat_and_survival",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": observation_size,
            "action_count": action_count,
            "architecture": "separate_actor_and_critic",
            "actor_initialization": (
                "stage_4_resume"
                if args.resume
                else "expanded_stage_3_actor"
                if args.stage3_checkpoint
                else "fresh"
            ),
            "critic_initialization": "stage_4_resume" if args.resume else "fresh",
        }
    )
    (model_directory / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )


def write_rollout_summaries(
    *,
    tf: Any,
    summary_writer: Any,
    metrics: dict[str, float],
    rollout: dict[str, np.ndarray],
    step: int,
) -> None:
    write_training_summaries(tf, summary_writer, metrics, rollout, step)
    with summary_writer.as_default():
        for name, value in {
            "combat/mean_damage_dealt_per_step": np.mean(rollout["damage_dealt"]),
            "combat/mean_damage_taken_per_step": np.mean(rollout["damage_taken"]),
            "combat/total_damage_taken": np.sum(rollout["damage_taken"]),
            "combat/bot_defeats": np.sum(rollout["bot_defeats"]),
            "combat/out_of_range_attacks": np.sum(rollout["invalid_attacks"]),
            "combat/attack_selections": np.sum(rollout["attack_selections"]),
            "combat/valid_attack_attempts": np.sum(
                rollout["valid_attack_attempts"]
            ),
            "combat/confirmed_hits": np.sum(rollout["confirmed_hits"]),
            "combat/target_tracking_failures": np.sum(
                rollout["target_tracking_failures"]
            ),
            "retreat/retreat_actions": np.sum(rollout["retreat_actions"]),
            "retreat/safe_area_actions": np.sum(rollout["safe_area_actions"]),
            "retreat/wait_actions": np.sum(rollout["wait_actions"]),
            "retreat/low_health_fraction": np.mean(rollout["low_health_steps"]),
            "retreat/mean_progress_reward": np.mean(rollout["retreat_rewards"]),
            "cover/covered_fraction": np.mean(rollout["cover_steps"]),
            "cover/entries": np.sum(rollout["cover_entries"]),
            "cover/total_reward": np.sum(rollout["cover_rewards"]),
            "outcome/survival_rewards": np.sum(rollout["survival_rewards"]),
            "outcome/death_penalties": np.sum(rollout["death_penalties"]),
            "positioning/mean_approach_reward": np.mean(
                rollout["approach_rewards"]
            ),
        }.items():
            tf.summary.scalar(name, value, step=step)
    summary_writer.flush()


def print_rollout(
    step: int,
    timesteps: int,
    rollout: dict[str, np.ndarray],
    metrics: dict[str, float],
) -> None:
    print(
        f"step={step:,}/{timesteps:,} "
        f"reward={np.mean(rollout['rewards']):+.3f} "
        f"dealt={np.sum(rollout['damage_dealt']):.1f} "
        f"taken={np.sum(rollout['damage_taken']):.1f} "
        f"deaths={int(np.sum(rollout['bot_defeats']))} "
        f"hits={int(np.sum(rollout['confirmed_hits']))} "
        f"invalid={int(np.sum(rollout['invalid_attacks']))} "
        f"retreat={int(np.sum(rollout['retreat_actions']))} "
        f"safe={int(np.sum(rollout['safe_area_actions']))} "
        f"cover_entries={int(np.sum(rollout['cover_entries']))} "
        f"covered={np.mean(rollout['cover_steps']):.1%} "
        f"low_health={np.mean(rollout['low_health_steps']):.1%} "
        f"survival_bonus={np.sum(rollout['survival_rewards']):.1f} "
        f"actor_epochs={int(metrics['actor_epochs_completed'])} "
        f"kl_stop={int(metrics['actor_early_stopped'])} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )
