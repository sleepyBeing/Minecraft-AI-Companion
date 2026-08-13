"""Configuration, TensorBoard, and terminal reporting for Stage Five."""

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
    for name in ("resume", "stage4_checkpoint"):
        configuration[name] = (
            str(configuration[name]) if configuration[name] else None
        )
    configuration.update(
        {
            "stage": 5,
            "task": "retreat_eat_recover_reengage",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": observation_size,
            "action_count": action_count,
            "duration_aware_discounting": True,
            "architecture": "separate_actor_and_critic",
            "actor_initialization": (
                "stage_5_resume"
                if args.resume
                else "expanded_stage_4_actor"
                if args.stage4_checkpoint
                else "fresh"
            ),
            "critic_initialization": "stage_5_resume" if args.resume else "fresh",
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
        values = {
            "combat/damage_dealt": np.sum(rollout["damage_dealt"]),
            "combat/damage_taken": np.sum(rollout["damage_taken"]),
            "combat/deaths": np.sum(rollout["bot_defeats"]),
            "combat/attack_actions": np.sum(rollout["attack_actions"]),
            "combat/invalid_attacks": np.sum(rollout["invalid_attacks"]),
            "retreat/actions": np.sum(rollout["retreat_actions"]),
            "retreat/safe_area_actions": np.sum(
                rollout["safe_area_actions"]
            ),
            "cover/confirmed_fraction": np.mean(rollout["cover_steps"]),
            "cover/raw_occluded_fraction": np.mean(
                rollout["cover_occluded_steps"]
            ),
            "recovery/eat_actions": np.sum(rollout["eat_actions"]),
            "recovery/successful_eats": np.sum(
                rollout["successful_eats"]
            ),
            "recovery/close_eat_attempts": np.sum(
                rollout["close_eat_attempts"]
            ),
            "recovery/unsafe_eat_attempts": np.sum(
                rollout["unsafe_eat_attempts"]
            ),
            "recovery/health_restored": np.sum(
                rollout["recovered_health"]
            ),
            "recovery/recovered_fraction": np.mean(
                rollout["has_recovered_steps"]
            ),
            "recovery/safe_to_eat_fraction": np.mean(
                rollout["safe_to_eat_steps"]
            ),
            "recovery/safe_eat_rewards": np.sum(
                rollout["safe_eat_reward"]
            ),
            "recovery/health_rewards": np.sum(
                rollout["health_recovery_reward"]
            ),
            "recovery/completion_rewards": np.sum(
                rollout["recovery_completion_reward"]
            ),
            "reengage/actions": np.sum(rollout["reengage_actions"]),
            "reengage/range_rewards": np.sum(rollout["reengage_reward"]),
            "outcome/recovery_victory_rewards": np.sum(
                rollout["recovery_victory_reward"]
            ),
            "outcome/recovery_victories": np.sum(rollout["successes"]),
            "outcome/survival_rewards": np.sum(rollout["survival_reward"]),
            "outcome/timeout_rewards": np.sum(rollout["timeout_reward"]),
            "outcome/death_penalties": np.sum(rollout["death_penalty"]),
            "timing/mean_action_duration_steps": np.mean(
                rollout["action_duration_steps"]
            ),
        }
        for name, value in values.items():
            tf.summary.scalar(name, value, step=step)
    summary_writer.flush()


def print_rollout(
    step: int,
    timesteps: int,
    rollout: dict[str, np.ndarray],
    metrics: dict[str, float],
) -> None:
    covered_eats = np.sum(rollout["safe_eat_reward"] > 0)
    open_eats = max(
        0,
        int(
            np.sum(rollout["successful_eats"])
            - np.sum(rollout["close_eat_attempts"])
            - covered_eats
        ),
    )
    print(
        f"step={step:,}/{timesteps:,} "
        f"reward={np.mean(rollout['rewards']):+.3f} "
        f"deaths={int(np.sum(rollout['bot_defeats']))} "
        f"eat={int(np.sum(rollout['eat_actions']))} "
        f"successful_eats={int(np.sum(rollout['successful_eats']))} "
        f"covered_eats={int(covered_eats)} "
        f"open_eats={open_eats} "
        f"close_eats={int(np.sum(rollout['close_eat_attempts']))} "
        f"recovered={np.mean(rollout['has_recovered_steps']):.1%} "
        f"reengage={int(np.sum(rollout['reengage_actions']))} "
        f"covered={np.mean(rollout['cover_steps']):.1%} "
        f"recovery_wins={int(np.sum(rollout['successes']))} "
        f"actor_epochs={int(metrics['actor_epochs_completed'])} "
        f"kl_stop={int(metrics['actor_early_stopped'])} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )
