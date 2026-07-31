"""Command-line entry point for TensorFlow PPO Stage Two training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
    StageTwoStationaryCombatEnv,
)
from agent.training.stage_2.stage_2_models import (
    build_stage_two_actor,
    build_stage_two_critic,
    transfer_legacy_stage_two_actor,
    transfer_stage_one_actor,
)
from agent.training.stage_2.stage_2_ppo import update_stage_two_ppo
from agent.training.stage_2.stage_2_preflight import run_live_combat_preflight
from agent.training.stage_2.stage_2_rollout import (
    StageTwoEpisodeCsvLogger,
    StageTwoEpisodeState,
    collect_stage_two_rollout,
)
from agent.training.stage_1.train_stage_1 import (
    import_tensorflow,
    positive_float,
    positive_integer,
    resolve_checkpoint,
    unit_interval,
    write_training_summaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMESTEPS = 150_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native TensorFlow PPO on Stage Two combat."
    )
    parser.add_argument("--timesteps", type=positive_integer, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--bridge-timeout", type=positive_float, default=15.0)
    parser.add_argument("--rollout-steps", type=positive_integer, default=1_024)
    parser.add_argument("--batch-size", type=positive_integer, default=64)

    actor = parser.add_argument_group("actor")
    actor.add_argument("--epochs", type=positive_integer, default=10)
    actor.add_argument("--learning-rate", type=positive_float, default=3e-4)
    actor.add_argument("--clip-ratio", type=positive_float, default=0.2)
    actor.add_argument(
        "--target-kl",
        type=positive_float,
        default=0.03,
        help=(
            "Stop the remaining actor epochs when an epoch's mean "
            "approximate KL exceeds this value."
        ),
    )
    actor.add_argument("--entropy-coefficient", type=float, default=0.05)
    actor.add_argument("--max-gradient-norm", type=positive_float, default=0.5)
    actor.add_argument("--hidden-size", type=positive_integer, default=128)

    critic = parser.add_argument_group("critic")
    critic.add_argument("--critic-epochs", type=positive_integer, default=10)
    critic.add_argument(
        "--critic-learning-rate", type=positive_float, default=3e-4
    )
    critic.add_argument(
        "--critic-hidden-size", type=positive_integer, default=128
    )
    critic.add_argument(
        "--critic-loss",
        choices=("huber", "mse"),
        default="huber",
        help="Stage Two critic loss; independent of Stage One.",
    )
    critic.add_argument(
        "--critic-huber-delta",
        type=positive_float,
        default=10.0,
        help="Huber transition point, used only with --critic-loss huber.",
    )
    critic.add_argument(
        "--critic-max-gradient-norm", type=positive_float, default=1.0
    )
    critic.add_argument(
        "--value-coefficient", type=positive_float, default=1.0
    )

    returns = parser.add_argument_group("returns")
    returns.add_argument("--gamma", type=unit_interval, default=0.99)
    returns.add_argument("--gae-lambda", type=unit_interval, default=0.95)

    parser.add_argument("--checkpoint-freq", type=positive_integer, default=10_000)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Separate Stage Two actor/critic checkpoint to restore.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Initialize only the Stage Two actor from Stage One.",
    )
    parser.add_argument(
        "--legacy-stage2-actor-checkpoint",
        type=Path,
        help=(
            "Initialize only the actor from an older combined Stage Two "
            "checkpoint; the independent critic starts fresh."
        ),
    )
    parser.add_argument(
        "--stage1-hidden-size",
        type=positive_integer,
        default=128,
        help="Hidden width used by the Stage One checkpoint.",
    )
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    _validate_arguments(args)

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    environment, source = _create_environment(args)
    run_name = args.run_name or datetime.now().strftime(
        "stage2_stationary_combat_%Y%m%d_%H%M%S"
    )
    model_directory = PROJECT_ROOT / "agent" / "models" / "stage_2" / run_name
    log_directory = PROJECT_ROOT / "agent" / "logs" / "stage_2" / run_name
    model_directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    observation_shape = environment.observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("Stage Two requires a one-dimensional observation space")
    observation_size = int(observation_shape[0])
    action_count = int(environment.action_space.n)

    actor = build_stage_two_actor(
        tf,
        observation_size=observation_size,
        action_count=action_count,
        hidden_size=args.hidden_size,
    )
    critic = build_stage_two_critic(
        tf,
        observation_size=observation_size,
        hidden_size=args.critic_hidden_size,
    )
    actor_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(args.critic_learning_rate)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    checkpoint = tf.train.Checkpoint(
        step=global_step,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor=actor,
        critic=critic,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(model_directory / "checkpoints"),
        max_to_keep=5,
    )

    _initialize_models(
        tf=tf,
        args=args,
        actor=actor,
        checkpoint=checkpoint,
        observation_size=observation_size,
        action_count=action_count,
    )
    _write_configuration(
        args=args,
        source=source,
        observation_size=observation_size,
        action_count=action_count,
        model_directory=model_directory,
    )

    summary_writer = tf.summary.create_file_writer(
        str(log_directory / "tensorboard")
    )
    episode_log = StageTwoEpisodeCsvLogger(log_directory / "episodes.csv")
    observation, _ = environment.reset(seed=args.seed)
    episode_state = StageTwoEpisodeState()
    next_checkpoint_step = (
        int(global_step.numpy()) // args.checkpoint_freq + 1
    ) * args.checkpoint_freq

    if source == "minecraft":
        minimum_hours = args.timesteps * environment.STEP_SECONDS / 3_600
        print(
            f"Live training requires at least {minimum_hours:.2f} hours for "
            f"{args.timesteps:,} actions at 10 actions/second."
        )
    print(f"Models: {model_directory}")
    print(f"Logs:   {log_directory}")

    try:
        while int(global_step.numpy()) < args.timesteps:
            remaining = args.timesteps - int(global_step.numpy())
            rollout_size = min(args.rollout_steps, remaining)
            rollout, observation, episode_state = collect_stage_two_rollout(
                tf=tf,
                actor=actor,
                critic=critic,
                environment=environment,
                initial_observation=observation,
                rollout_size=rollout_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                episode_state=episode_state,
                episode_logger=episode_log,
                summary_writer=summary_writer,
            )
            metrics = update_stage_two_ppo(
                tf=tf,
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                rollout=rollout,
                actor_epochs=args.epochs,
                critic_epochs=args.critic_epochs,
                batch_size=args.batch_size,
                clip_ratio=args.clip_ratio,
                target_kl=args.target_kl,
                entropy_coefficient=args.entropy_coefficient,
                value_coefficient=args.value_coefficient,
                actor_max_gradient_norm=args.max_gradient_norm,
                critic_max_gradient_norm=args.critic_max_gradient_norm,
                critic_loss_name=args.critic_loss,
                critic_huber_delta=args.critic_huber_delta,
            )
            global_step.assign_add(rollout_size)
            step = int(global_step.numpy())
            _write_rollout_summaries(
                tf=tf,
                summary_writer=summary_writer,
                metrics=metrics,
                rollout=rollout,
                step=step,
            )

            if step >= next_checkpoint_step:
                saved_path = checkpoint_manager.save(checkpoint_number=step)
                print(f"Checkpoint saved: {saved_path}")
                next_checkpoint_step += args.checkpoint_freq
            _print_rollout(step, args.timesteps, rollout, metrics)
    except KeyboardInterrupt:
        saved_path = checkpoint_manager.save(
            checkpoint_number=int(global_step.numpy())
        )
        print(f"\nTraining interrupted; checkpoint saved: {saved_path}")
    except Exception:
        saved_path = checkpoint_manager.save(
            checkpoint_number=int(global_step.numpy())
        )
        print(f"\nTraining failed; emergency checkpoint saved: {saved_path}")
        raise
    else:
        saved_path = checkpoint_manager.save(
            checkpoint_number=int(global_step.numpy())
        )
        actor.save_weights(model_directory / "final_actor.weights.h5")
        critic.save_weights(model_directory / "final_critic.weights.h5")
        print(f"Training complete; checkpoint saved: {saved_path}")
    finally:
        episode_log.close()
        summary_writer.flush()
        summary_writer.close()
        environment.close()


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size > args.rollout_steps:
        raise ValueError("--batch-size cannot exceed --rollout-steps")
    initialization_sources = (
        bool(args.resume),
        bool(args.stage1_checkpoint),
        bool(args.legacy_stage2_actor_checkpoint),
    )
    if sum(initialization_sources) > 1:
        raise ValueError(
            "--resume, --stage1-checkpoint, and "
            "--legacy-stage2-actor-checkpoint are mutually exclusive"
        )
    if args.stage1_checkpoint and args.stage1_hidden_size != args.hidden_size:
        raise ValueError(
            "Stage One transfer requires matching hidden sizes"
        )


def _create_environment(
    args: argparse.Namespace,
) -> tuple[StageTwoStationaryCombatEnv, str]:
    if args.simulation:
        return StageTwoStationaryCombatEnv(), "simulation"
    environment = LiveStageTwoStationaryCombatEnv(
        bridge_url=args.bridge_url,
        bridge_timeout=args.bridge_timeout,
    )
    environment.bridge.connect()
    run_live_combat_preflight(environment)
    return environment, "minecraft"


def _initialize_models(
    *,
    tf: object,
    args: argparse.Namespace,
    actor: object,
    checkpoint: object,
    observation_size: int,
    action_count: int,
) -> None:
    if args.resume:
        restored_path = resolve_checkpoint(tf, args.resume)
        names = {name for name, _ in tf.train.list_variables(restored_path)}
        if not (
            any(name.startswith("actor_optimizer/") for name in names)
            and any(name.startswith("critic_optimizer/") for name in names)
        ):
            raise ValueError(
                "--resume requires a separate actor/critic checkpoint. Use "
                "--legacy-stage2-actor-checkpoint for an older combined model."
            )
        checkpoint.restore(restored_path).expect_partial()
        print(f"Restored Stage Two checkpoint: {restored_path}")
    elif args.stage1_checkpoint:
        source_path = resolve_checkpoint(tf, args.stage1_checkpoint)
        transfer_stage_one_actor(
            tf=tf,
            stage_two_actor=actor,
            checkpoint_path=source_path,
            hidden_size=args.stage1_hidden_size,
        )
        print(f"Initialized Stage Two actor from Stage One: {source_path}")
    elif args.legacy_stage2_actor_checkpoint:
        source_path = resolve_checkpoint(tf, args.legacy_stage2_actor_checkpoint)
        transfer_legacy_stage_two_actor(
            tf=tf,
            stage_two_actor=actor,
            checkpoint_path=source_path,
            hidden_size=args.hidden_size,
            observation_size=observation_size,
            action_count=action_count,
        )
        print(
            "Initialized actor from legacy Stage Two checkpoint; "
            f"critic started fresh: {source_path}"
        )


def _write_configuration(
    *,
    args: argparse.Namespace,
    source: str,
    observation_size: int,
    action_count: int,
    model_directory: Path,
) -> None:
    configuration = vars(args).copy()
    for name in (
        "resume",
        "stage1_checkpoint",
        "legacy_stage2_actor_checkpoint",
    ):
        configuration[name] = (
            str(configuration[name]) if configuration[name] else None
        )
    configuration.update(
        {
            "stage": 2,
            "task": "stationary_zombie_combat",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": observation_size,
            "action_count": action_count,
            "architecture": "separate_actor_and_critic",
        }
    )
    (model_directory / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )


def _write_rollout_summaries(
    *,
    tf: object,
    summary_writer: object,
    metrics: dict[str, float],
    rollout: dict[str, np.ndarray],
    step: int,
) -> None:
    write_training_summaries(tf, summary_writer, metrics, rollout, step)
    with summary_writer.as_default():
        for name, value in {
            "combat/mean_damage_per_step": np.mean(rollout["damage_dealt"]),
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
            "combat/cooldown_blocked_attacks": np.sum(
                rollout["cooldown_blocked_attacks"]
            ),
            "combat/confirmed_hits": np.sum(rollout["confirmed_hits"]),
            "positioning/mean_approach_reward": np.mean(
                rollout["approach_rewards"]
            ),
        }.items():
            tf.summary.scalar(name, value, step=step)
    summary_writer.flush()


def _print_rollout(
    step: int,
    timesteps: int,
    rollout: dict[str, np.ndarray],
    metrics: dict[str, float],
) -> None:
    print(
        f"step={step:,}/{timesteps:,} "
        f"reward={np.mean(rollout['rewards']):+.3f} "
        f"damage={np.sum(rollout['damage_dealt']):.1f} "
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


if __name__ == "__main__":
    main()
