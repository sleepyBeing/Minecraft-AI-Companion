"""Train native TensorFlow PPO on stage-two stationary-zombie combat.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agent.stages.stage_2_stationary_combat import (
    LiveStageTwoStationaryCombatEnv,
    StageTwoStationaryCombatEnv,
)
from agent.training.train_stage_1 import (
    build_actor_critic,
    import_tensorflow,
    positive_float,
    positive_integer,
    resolve_checkpoint,
    unit_interval,
    update_ppo,
    write_training_summaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMESTEPS = 150_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native TensorFlow PPO on stage-two stationary combat."
    )
    parser.add_argument("--timesteps", type=positive_integer, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--bridge-timeout", type=positive_float, default=15.0)
    parser.add_argument("--rollout-steps", type=positive_integer, default=1_024)
    parser.add_argument("--batch-size", type=positive_integer, default=64)
    parser.add_argument("--epochs", type=positive_integer, default=10)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--gamma", type=unit_interval, default=0.99)
    parser.add_argument("--gae-lambda", type=unit_interval, default=0.95)
    parser.add_argument("--clip-ratio", type=positive_float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.05)
    parser.add_argument("--value-coefficient", type=positive_float, default=0.5)
    parser.add_argument("--max-gradient-norm", type=positive_float, default=0.5)
    parser.add_argument("--hidden-size", type=positive_integer, default=128)
    parser.add_argument("--checkpoint-freq", type=positive_integer, default=10_000)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Stage-two checkpoint directory or checkpoint prefix to restore.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help=(
            "Optionally initialize compatible layers from a stage-one checkpoint. "
            "Use this only when starting a new stage-two run."
        ),
    )
    parser.add_argument(
        "--stage1-hidden-size",
        type=positive_integer,
        default=128,
        help="Hidden width used by the stage-one checkpoint.",
    )
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size > args.rollout_steps:
        raise ValueError("--batch-size cannot exceed --rollout-steps")
    if args.resume and args.stage1_checkpoint:
        raise ValueError("--resume and --stage1-checkpoint cannot be used together")
    if args.stage1_checkpoint and args.stage1_hidden_size != args.hidden_size:
        raise ValueError(
            "Stage-one transfer requires matching --stage1-hidden-size and --hidden-size"
        )

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or datetime.now().strftime(
        "stage2_stationary_combat_%Y%m%d_%H%M%S"
    )
    model_directory = PROJECT_ROOT / "agent" / "models" / "stage_2" / run_name
    log_directory = PROJECT_ROOT / "agent" / "logs" / "stage_2" / run_name
    model_directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    environment: StageTwoStationaryCombatEnv
    if args.simulation:
        environment = StageTwoStationaryCombatEnv()
        source = "simulation"
    else:
        live_environment = LiveStageTwoStationaryCombatEnv(
            bridge_url=args.bridge_url,
            bridge_timeout=args.bridge_timeout,
        )
        live_environment.bridge.connect()
        environment = live_environment
        source = "minecraft"

    observation_shape = environment.observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("Stage two requires a one-dimensional observation space")

    observation_size = int(observation_shape[0])
    action_count = int(environment.action_space.n)
    model = build_actor_critic(
        tf,
        observation_size=observation_size,
        action_count=action_count,
        hidden_size=args.hidden_size,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    checkpoint = tf.train.Checkpoint(
        step=global_step,
        optimizer=optimizer,
        model=model,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(model_directory / "checkpoints"),
        max_to_keep=5,
    )

    if args.resume:
        restored_path = resolve_checkpoint(tf, args.resume)
        checkpoint.restore(restored_path).expect_partial()
        print(f"Restored stage-two checkpoint: {restored_path}")
    elif args.stage1_checkpoint:
        source_path = resolve_checkpoint(tf, args.stage1_checkpoint)
        transfer_stage_one_weights(
            tf=tf,
            stage_two_model=model,
            checkpoint_path=source_path,
            hidden_size=args.stage1_hidden_size,
        )
        print(f"Initialized compatible layers from stage one: {source_path}")

    configuration = vars(args).copy()
    configuration["resume"] = str(args.resume) if args.resume else None
    configuration["stage1_checkpoint"] = (
        str(args.stage1_checkpoint) if args.stage1_checkpoint else None
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
        }
    )
    (model_directory / "training_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    summary_writer = tf.summary.create_file_writer(str(log_directory / "tensorboard"))
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
                model=model,
                environment=environment,
                initial_observation=observation,
                rollout_size=rollout_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                episode_state=episode_state,
                episode_logger=episode_log,
                summary_writer=summary_writer,
            )
            metrics = update_ppo(
                tf=tf,
                model=model,
                optimizer=optimizer,
                rollout=rollout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                clip_ratio=args.clip_ratio,
                entropy_coefficient=args.entropy_coefficient,
                value_coefficient=args.value_coefficient,
                max_gradient_norm=args.max_gradient_norm,
            )
            global_step.assign_add(rollout_size)
            step = int(global_step.numpy())
            write_training_summaries(tf, summary_writer, metrics, rollout, step)
            with summary_writer.as_default():
                tf.summary.scalar(
                    "combat/mean_damage_per_step",
                    np.mean(rollout["damage_dealt"]),
                    step=step,
                )
                tf.summary.scalar(
                    "combat/out_of_range_attacks",
                    np.sum(rollout["invalid_attacks"]),
                    step=step,
                )
            summary_writer.flush()

            if step >= next_checkpoint_step:
                saved_path = checkpoint_manager.save(checkpoint_number=step)
                print(f"Checkpoint saved: {saved_path}")
                next_checkpoint_step += args.checkpoint_freq

            print(
                f"step={step:,}/{args.timesteps:,} "
                f"reward={np.mean(rollout['rewards']):+.3f} "
                f"damage={np.sum(rollout['damage_dealt']):.1f} "
                f"invalid_attacks={int(np.sum(rollout['invalid_attacks']))} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f}"
            )
    except KeyboardInterrupt:
        saved_path = checkpoint_manager.save(checkpoint_number=int(global_step.numpy()))
        print(f"\nTraining interrupted; checkpoint saved: {saved_path}")
    except Exception:
        saved_path = checkpoint_manager.save(checkpoint_number=int(global_step.numpy()))
        print(f"\nTraining failed; emergency checkpoint saved: {saved_path}")
        raise
    else:
        saved_path = checkpoint_manager.save(checkpoint_number=int(global_step.numpy()))
        model.save_weights(model_directory / "final_model.weights.h5")
        print(f"Training complete; checkpoint saved: {saved_path}")
    finally:
        episode_log.close()
        summary_writer.flush()
        summary_writer.close()
        environment.close()


class StageTwoEpisodeState:
    def __init__(self) -> None:
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_number = 0
        self.damage_dealt = 0.0
        self.invalid_attacks = 0


def collect_stage_two_rollout(
    *,
    tf: Any,
    model: Any,
    environment: StageTwoStationaryCombatEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_state: StageTwoEpisodeState,
    episode_logger: "StageTwoEpisodeCsvLogger",
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, StageTwoEpisodeState]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    values: list[float] = []
    log_probabilities: list[float] = []
    next_values: list[float] = []
    terminated_flags: list[bool] = []
    episode_done_flags: list[bool] = []
    damages: list[float] = []
    invalid_attacks: list[float] = []
    observation = initial_observation

    for _ in range(rollout_size):
        observation_tensor = tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
        logits, value_tensor = model(observation_tensor, training=False)
        action = int(tf.random.categorical(logits, 1)[0, 0].numpy())
        log_probability = float(tf.nn.log_softmax(logits)[0, action].numpy())
        value = float(value_tensor[0, 0].numpy())

        next_observation, reward, terminated, truncated, info = environment.step(action)
        next_value_tensor = model(
            tf.convert_to_tensor(next_observation[None, :], dtype=tf.float32),
            training=False,
        )[1]
        next_value = float(next_value_tensor[0, 0].numpy())
        episode_done = terminated or truncated
        damage = float(info["damage_dealt"])
        invalid_attack = bool(info["invalid_attack"])

        observations.append(observation.copy())
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probabilities.append(log_probability)
        next_values.append(next_value)
        terminated_flags.append(terminated)
        episode_done_flags.append(episode_done)
        damages.append(damage)
        invalid_attacks.append(float(invalid_attack))

        episode_state.episode_return += reward
        episode_state.episode_length += 1
        episode_state.damage_dealt += damage
        episode_state.invalid_attacks += int(invalid_attack)
        observation = next_observation

        if episode_done:
            episode_state.episode_number += 1
            success = bool(info["success"])
            episode_logger.write(
                episode=episode_state.episode_number,
                episode_return=episode_state.episode_return,
                length=episode_state.episode_length,
                success=success,
                final_distance=float(info["distance_to_target"]),
                target_health=float(info["target_health"]),
                damage_dealt=episode_state.damage_dealt,
                invalid_attacks=episode_state.invalid_attacks,
            )
            with summary_writer.as_default():
                tf.summary.scalar(
                    "episode/return",
                    episode_state.episode_return,
                    step=episode_state.episode_number,
                )
                tf.summary.scalar(
                    "episode/length",
                    episode_state.episode_length,
                    step=episode_state.episode_number,
                )
                tf.summary.scalar(
                    "episode/success",
                    float(success),
                    step=episode_state.episode_number,
                )
                tf.summary.scalar(
                    "episode/damage_dealt",
                    episode_state.damage_dealt,
                    step=episode_state.episode_number,
                )
                tf.summary.scalar(
                    "episode/invalid_attacks",
                    episode_state.invalid_attacks,
                    step=episode_state.episode_number,
                )
            observation, _ = environment.reset()
            episode_state.episode_return = 0.0
            episode_state.episode_length = 0
            episode_state.damage_dealt = 0.0
            episode_state.invalid_attacks = 0

    reward_array = np.asarray(rewards, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32)
    next_value_array = np.asarray(next_values, dtype=np.float32)
    terminated_array = np.asarray(terminated_flags, dtype=np.float32)
    done_array = np.asarray(episode_done_flags, dtype=np.float32)
    deltas = (
        reward_array
        + gamma * next_value_array * (1.0 - terminated_array)
        - value_array
    )
    advantages = np.zeros_like(reward_array)
    gae = 0.0
    for index in range(rollout_size - 1, -1, -1):
        gae = deltas[index] + gamma * gae_lambda * (1.0 - done_array[index]) * gae
        advantages[index] = gae
    returns = advantages + value_array
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rollout = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int32),
        "rewards": reward_array,
        "values": value_array,
        "log_probabilities": np.asarray(log_probabilities, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "damage_dealt": np.asarray(damages, dtype=np.float32),
        "invalid_attacks": np.asarray(invalid_attacks, dtype=np.float32),
    }
    return rollout, observation, episode_state


class StageTwoEpisodeCsvLogger:
    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "episode",
                "return",
                "length",
                "success",
                "final_distance",
                "target_health",
                "damage_dealt",
                "invalid_attacks",
            ]
        )

    def write(
        self,
        *,
        episode: int,
        episode_return: float,
        length: int,
        success: bool,
        final_distance: float,
        target_health: float,
        damage_dealt: float,
        invalid_attacks: int,
    ) -> None:
        self._writer.writerow(
            [
                episode,
                episode_return,
                length,
                int(success),
                final_distance,
                target_health,
                damage_dealt,
                invalid_attacks,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def transfer_stage_one_weights(
    *,
    tf: Any,
    stage_two_model: Any,
    checkpoint_path: str,
    hidden_size: int,
) -> None:
    """Copy shared stage-one behavior while preserving new stage-two weights."""

    stage_one_model = build_actor_critic(
        tf,
        observation_size=10,
        action_count=7,
        hidden_size=hidden_size,
    )
    source_checkpoint = tf.train.Checkpoint(model=stage_one_model)
    status = source_checkpoint.restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    source_hidden_one = stage_one_model.layers[1]
    source_hidden_two = stage_one_model.layers[2]
    source_policy = stage_one_model.layers[3]
    source_value = stage_one_model.layers[4]
    target_hidden_one = stage_two_model.layers[1]
    target_hidden_two = stage_two_model.layers[2]
    target_policy = stage_two_model.layers[3]
    target_value = stage_two_model.layers[4]

    source_kernel, source_bias = source_hidden_one.get_weights()
    target_kernel, _ = target_hidden_one.get_weights()
    target_kernel[: source_kernel.shape[0], :] = source_kernel
    target_hidden_one.set_weights([target_kernel, source_bias])
    target_hidden_two.set_weights(source_hidden_two.get_weights())

    source_policy_kernel, source_policy_bias = source_policy.get_weights()
    target_policy_kernel, target_policy_bias = target_policy.get_weights()
    target_policy_kernel[:, : source_policy_kernel.shape[1]] = source_policy_kernel
    target_policy_bias[: source_policy_bias.shape[0]] = source_policy_bias
    target_policy.set_weights([target_policy_kernel, target_policy_bias])
    target_value.set_weights(source_value.get_weights())


if __name__ == "__main__":
    main()
