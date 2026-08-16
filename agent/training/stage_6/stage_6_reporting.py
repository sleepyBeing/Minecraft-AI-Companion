"""Configuration, TensorBoard, and terminal reporting for Stage Six."""

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
    for name in ("resume", "stage5_checkpoint"):
        configuration[name] = (
            str(configuration[name]) if configuration[name] else None
        )
    configuration.update(
        {
            "stage": 6,
            "task": "protect_stationary_npc",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": observation_size,
            "action_count": action_count,
            "duration_aware_discounting": True,
            "architecture": "separate_actor_and_critic",
            "actor_initialization": (
                "stage_6_resume"
                if args.resume
                else "expanded_stage_5_actor"
                if args.stage5_checkpoint
                else "fresh"
            ),
            "critic_initialization": "stage_6_resume" if args.resume else "fresh",
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
            "combat/bot_deaths": np.sum(rollout["bot_defeats"]),
            "combat/attack_actions": np.sum(rollout["attack_actions"]),
            "combat/invalid_attacks": np.sum(rollout["invalid_attacks"]),
            "combat/bot_damage_penalties": np.sum(
                rollout["bot_damage_penalty"]
            ),
            "retreat/actions": np.sum(rollout["retreat_actions"]),
            "retreat/safe_area_actions": np.sum(
                rollout["safe_area_actions"]
            ),
            "cover/confirmed_fraction": np.mean(rollout["cover_steps"]),
            "cover/occluded_fraction": np.mean(
                rollout["cover_occluded_steps"]
            ),
            "recovery/eat_actions": np.sum(rollout["eat_actions"]),
            "recovery/successful_eats": np.sum(
                rollout["successful_eats"]
            ),
            "recovery/health_restored": np.sum(
                rollout["recovered_health"]
            ),
            "recovery/recovered_fraction": np.mean(
                rollout["has_recovered_steps"]
            ),
            "reengage/actions": np.sum(rollout["reengage_actions"]),
            "reengage/range_rewards": np.sum(rollout["reengage_reward"]),
            "outcome/protected_victories": np.sum(rollout["successes"]),
            "outcome/bot_death_penalties": np.sum(rollout["death_penalty"]),
            "timing/mean_action_duration_steps": np.mean(
                rollout["action_duration_steps"]
            ),
            "protection/npc_damage_taken": np.sum(rollout["npc_damage_taken"]),
            "protection/npc_damage_penalties": np.sum(
                rollout["npc_damage_penalty"]
            ),
            "protection/npc_deaths": np.sum(rollout["npc_defeats"]),
            "protection/npc_survival_rewards": np.sum(
                rollout["npc_survival_reward"]
            ),
            "protection/survived_episodes": np.sum(
                rollout["npc_survived_episode_flags"]
            ),
            "protection/npc_alive_fraction": np.mean(rollout["npc_alive_steps"]),
            "protection/npc_under_attack_fraction": np.mean(
                rollout["npc_under_attack_steps"]
            ),
            "protection/mean_bot_npc_distance": np.mean(
                rollout["bot_npc_distance"]
            ),
            "protection/mean_enemy_npc_distance": np.mean(
                rollout["enemy_npc_distance"]
            ),
            "protection/mean_interaction_distance": np.mean(
                rollout["interaction_distance"]
            ),
            "protection/stray_penalties": np.sum(rollout["stray_penalty"]),
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
    print(
        f"step={step:,}/{timesteps:,} "
        f"reward={np.mean(rollout['rewards']):+.3f} "
        f"npc_damage={np.sum(rollout['npc_damage_taken']):.1f} "
        f"npc_deaths={int(np.sum(rollout['npc_defeats']))} "
        f"npc_survived={int(np.sum(rollout['npc_survived_episode_flags']))} "
        f"zombie_damage={np.sum(rollout['damage_dealt']):.1f} "
        f"bot_damage={np.sum(rollout['damage_taken']):.1f} "
        f"bot_deaths={int(np.sum(rollout['bot_defeats']))} "
        f"attack={int(np.sum(rollout['attack_actions']))} "
        f"eat={int(np.sum(rollout['eat_actions']))} "
        f"reengage={int(np.sum(rollout['reengage_actions']))} "
        f"stray_penalty={np.sum(rollout['stray_penalty']):.1f} "
        f"actor_epochs={int(metrics['actor_epochs_completed'])} "
        f"kl_stop={int(metrics['actor_early_stopped'])} "
        f"policy_loss={metrics['policy_loss']:.4f} "
        f"value_loss={metrics['value_loss']:.4f}"
    )
