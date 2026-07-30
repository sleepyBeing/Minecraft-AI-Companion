"""Train a native TensorFlow PPO agent on stage-one positioning.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agent.environment import BasicPositioningEnv
from agent.stages.stage_1_positioning import LiveBasicPositioningEnv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMESTEPS = 100_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native TensorFlow PPO on stage-one positioning."
    )
    parser.add_argument("--timesteps", type=positive_integer, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use the fast local simulation instead of the Minecraft bot.",
    )
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
        help="Checkpoint directory or TensorFlow checkpoint prefix to restore.",
    )
    parser.add_argument("--run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size > args.rollout_steps:
        raise ValueError("--batch-size cannot exceed --rollout-steps")

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("stage1_positioning_%Y%m%d_%H%M%S")
    model_directory = PROJECT_ROOT / "agent" / "models" / "stage_1" / run_name
    log_directory = PROJECT_ROOT / "agent" / "logs" / "stage_1" / run_name
    model_directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    environment: BasicPositioningEnv
    if args.simulation:
        environment = BasicPositioningEnv()
        source = "simulation"
    else:
        live_environment = LiveBasicPositioningEnv(
            bridge_url=args.bridge_url,
            bridge_timeout=args.bridge_timeout,
        )
        live_environment.bridge.connect()
        environment = live_environment
        source = "minecraft"

    observation_shape = environment.observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("Stage one requires a one-dimensional observation space")

    model = build_actor_critic(
        tf,
        observation_size=int(observation_shape[0]),
        action_count=int(environment.action_space.n),
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
        print(f"Restored TensorFlow PPO checkpoint: {restored_path}")

    configuration = vars(args).copy()
    configuration["resume"] = str(args.resume) if args.resume else None
    configuration.update(
        {
            "stage": 1,
            "task": "basic_positioning",
            "source": source,
            "algorithm": "PPO",
            "framework": "TensorFlow",
            "observation_size": int(observation_shape[0]),
            "action_count": int(environment.action_space.n),
        }
    )
    (model_directory / "training_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    summary_writer = tf.summary.create_file_writer(str(log_directory / "tensorboard"))
    episode_log = EpisodeCsvLogger(log_directory / "episodes.csv")
    observation, _ = environment.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_number = 0
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
            rollout, observation, episode_state = collect_rollout(
                tf=tf,
                model=model,
                environment=environment,
                initial_observation=observation,
                rollout_size=rollout_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                episode_return=episode_return,
                episode_length=episode_length,
                episode_number=episode_number,
                episode_logger=episode_log,
                summary_writer=summary_writer,
            )
            episode_return, episode_length, episode_number = episode_state

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

            if step >= next_checkpoint_step:
                saved_path = checkpoint_manager.save(checkpoint_number=step)
                print(f"Checkpoint saved: {saved_path}")
                next_checkpoint_step += args.checkpoint_freq

            print(
                f"step={step:,}/{args.timesteps:,} "
                f"reward={np.mean(rollout['rewards']):+.3f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )
    except KeyboardInterrupt:
        saved_path = checkpoint_manager.save(checkpoint_number=int(global_step.numpy()))
        print(f"\nTraining interrupted; checkpoint saved: {saved_path}")
    else:
        saved_path = checkpoint_manager.save(checkpoint_number=int(global_step.numpy()))
        model.save_weights(model_directory / "final_model.weights.h5")
        print(f"Training complete; checkpoint saved: {saved_path}")
    finally:
        episode_log.close()
        summary_writer.flush()
        summary_writer.close()
        environment.close()


def build_actor_critic(
    tf: Any,
    *,
    observation_size: int,
    action_count: int,
    hidden_size: int,
) -> Any:
    inputs = tf.keras.Input(shape=(observation_size,), dtype=tf.float32)
    hidden = tf.keras.layers.Dense(hidden_size, activation="tanh")(inputs)
    hidden = tf.keras.layers.Dense(hidden_size, activation="tanh")(hidden)
    # policy_logits outputs probabilities for each action, value outputs a single scalar value that represents the expected return
    policy_logits = tf.keras.layers.Dense(action_count, name="policy_logits")(hidden)
    value = tf.keras.layers.Dense(1, name="value")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=(policy_logits, value))


def collect_rollout(
    *,
    tf: Any,
    model: Any,
    environment: BasicPositioningEnv,
    initial_observation: np.ndarray,
    rollout_size: int,
    gamma: float,
    gae_lambda: float,
    episode_return: float,
    episode_length: int,
    episode_number: int,
    episode_logger: "EpisodeCsvLogger",
    summary_writer: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[float, int, int]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    values: list[float] = []
    log_probabilities: list[float] = []
    next_values: list[float] = []
    terminated_flags: list[bool] = []
    episode_done_flags: list[bool] = []
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

        observations.append(observation.copy())
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probabilities.append(log_probability)
        next_values.append(next_value)
        terminated_flags.append(terminated)
        episode_done_flags.append(episode_done)

        episode_return += reward
        episode_length += 1
        observation = next_observation

        if episode_done:
            episode_number += 1
            success = bool(info.get("success", False))
            episode_logger.write(
                episode_number,
                episode_return,
                episode_length,
                success,
                float(info["distance_to_target"]),
            )
            with summary_writer.as_default():
                tf.summary.scalar("episode/return", episode_return, step=episode_number)
                tf.summary.scalar("episode/length", episode_length, step=episode_number)
                tf.summary.scalar("episode/success", float(success), step=episode_number)
            observation, _ = environment.reset()
            episode_return = 0.0
            episode_length = 0

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
    }
    return rollout, observation, (episode_return, episode_length, episode_number)


def update_ppo(
    *,
    tf: Any,
    model: Any,
    optimizer: Any,
    rollout: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    entropy_coefficient: float,
    value_coefficient: float,
    max_gradient_norm: float,
) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approximate_kl": [],
        "clip_fraction": [],
    }
    sample_count = len(rollout["actions"])

    for _ in range(epochs):
        shuffled_indices = np.random.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            indices = shuffled_indices[start : start + batch_size]
            observations = tf.convert_to_tensor(rollout["observations"][indices])
            actions = tf.convert_to_tensor(rollout["actions"][indices])
            old_log_probabilities = tf.convert_to_tensor(
                rollout["log_probabilities"][indices]
            )
            old_values = tf.convert_to_tensor(rollout["values"][indices])
            advantages = tf.convert_to_tensor(rollout["advantages"][indices])
            returns = tf.convert_to_tensor(rollout["returns"][indices])

            with tf.GradientTape() as tape:
                logits, values = model(observations, training=True)
                values = tf.squeeze(values, axis=1)
                all_log_probabilities = tf.nn.log_softmax(logits)
                action_indices = tf.stack(
                    [tf.range(tf.shape(actions)[0]), actions],
                    axis=1,
                )
                new_log_probabilities = tf.gather_nd(
                    all_log_probabilities, action_indices
                )
                ratios = tf.exp(new_log_probabilities - old_log_probabilities)
                unclipped_objective = ratios * advantages
                clipped_objective = (
                    tf.clip_by_value(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
                    * advantages
                )
                policy_loss = -tf.reduce_mean(
                    tf.minimum(unclipped_objective, clipped_objective)
                )

                clipped_values = old_values + tf.clip_by_value(
                    values - old_values, -clip_ratio, clip_ratio
                )
                value_loss = 0.5 * tf.reduce_mean(
                    tf.maximum(
                        tf.square(values - returns),
                        tf.square(clipped_values - returns),
                    )
                )
                probabilities = tf.nn.softmax(logits)
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(probabilities * all_log_probabilities, axis=1)
                )
                total_loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    - entropy_coefficient * entropy
                )

            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            approximate_kl = tf.reduce_mean(
                old_log_probabilities - new_log_probabilities
            )
            clip_fraction = tf.reduce_mean(
                tf.cast(tf.abs(ratios - 1.0) > clip_ratio, tf.float32)
            )
            for name, value in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
            }.items():
                metric_values[name].append(float(value.numpy()))

    return {
        name: float(np.mean(values))
        for name, values in metric_values.items()
    }


def write_training_summaries(
    tf: Any,
    writer: Any,
    metrics: dict[str, float],
    rollout: dict[str, np.ndarray],
    step: int,
) -> None:
    predicted_values = rollout["values"]
    returns = rollout["returns"]
    return_variance = np.var(returns)
    explained_variance = (
        float("nan")
        if return_variance == 0
        else float(1.0 - np.var(returns - predicted_values) / return_variance)
    )
    with writer.as_default():
        for name, value in metrics.items():
            tf.summary.scalar(f"train/{name}", value, step=step)
        tf.summary.scalar("rollout/mean_reward", np.mean(rollout["rewards"]), step=step)
        tf.summary.scalar("train/explained_variance", explained_variance, step=step)
    writer.flush()


class EpisodeCsvLogger:
    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["episode", "return", "length", "success", "final_distance"]
        )

    def write(
        self,
        episode: int,
        episode_return: float,
        length: int,
        success: bool,
        final_distance: float,
    ) -> None:
        self._writer.writerow(
            [episode, episode_return, length, int(success), final_distance]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def resolve_checkpoint(tf: Any, path: Path) -> str:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        checkpoint_path = tf.train.latest_checkpoint(str(candidate))
        if not checkpoint_path and (candidate / "checkpoints").is_dir():
            checkpoint_path = tf.train.latest_checkpoint(str(candidate / "checkpoints"))
        if checkpoint_path:
            return checkpoint_path
    candidate_string = str(candidate)
    if Path(candidate_string + ".index").is_file():
        return candidate_string
    raise FileNotFoundError(f"TensorFlow checkpoint not found: {path}")


def import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TensorFlow is not installed. It has no Python 3.14 build; create "
            "this project virtual environment with Python 3.12 or 3.13, then "
            "run: python -m pip install -r requirements.txt"
        ) from error
    return tf


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


if __name__ == "__main__":
    main()
