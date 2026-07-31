"""Configuration, TensorBoard, and terminal reporting for Stage Three."""

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
    for name in ("resume", "stage2_checkpoint"):
        configuration[name] = (
            str(configuration[name]) if configuration[name] else None
        )
    configuration.update(
        {
            "stage": 3,
            "task": "moving_zombie_combat",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": observation_size,
            "action_count": action_count,
            "architecture": "separate_actor_and_critic",
            "critic_initialization": "fresh_unless_resuming_stage_3",
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
            "combat/target_tracking_failures": np.sum(
                rollout["target_tracking_failures"]
            ),
            "combat/movement_settle_failures": np.sum(
                rollout["movement_settle_failures"]
            ),
            "combat/attack_selections": np.sum(rollout["attack_selections"]),
            "combat/valid_attack_attempts": np.sum(
                rollout["valid_attack_attempts"]
            ),
            "combat/confirmed_hits": np.sum(rollout["confirmed_hits"]),
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
        f"damage_dealt={np.sum(rollout['damage_dealt']):.1f} "
        f"damage_taken={np.sum(rollout['damage_taken']):.1f} "
        f"bot_deaths={int(np.sum(rollout['bot_defeats']))} "
        f"attacks={int(np.sum(rollout['attack_selections']))} "
        f"valid={int(np.sum(rollout['valid_attack_attempts']))} "
        f"hits={int(np.sum(rollout['confirmed_hits']))} "
        f"invalid_attacks={int(np.sum(rollout['invalid_attacks']))} "
        f"tracking_failures={int(np.sum(rollout['target_tracking_failures']))} "
        f"settle_failures={int(np.sum(rollout['movement_settle_failures']))} "
        f"actor_epochs={int(metrics['actor_epochs_completed'])} "
        f"kl_stop={int(metrics['actor_early_stopped'])} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )
