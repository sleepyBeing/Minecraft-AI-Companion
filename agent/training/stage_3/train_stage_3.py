"""Command-line entry point for TensorFlow PPO Stage Three training."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agent.training.stage_1.train_stage_1 import import_tensorflow
from agent.training.stage_2.stage_2_models import (
    build_stage_two_actor,
    build_stage_two_critic,
)
from agent.training.stage_2.stage_2_ppo import update_stage_two_ppo
from agent.training.stage_2.stage_2_rollout import (
    StageTwoEpisodeCsvLogger as StageThreeEpisodeCsvLogger,
    StageTwoEpisodeState as StageThreeEpisodeState,
    collect_stage_two_rollout as collect_stage_three_rollout,
)
from agent.training.stage_3.stage_3_config import (
    PROJECT_ROOT,
    parse_arguments,
    validate_arguments,
)
from agent.training.stage_3.stage_3_models import initialize_models
from agent.training.stage_3.stage_3_reporting import (
    print_rollout,
    write_configuration,
    write_rollout_summaries,
)
from agent.training.stage_3.stage_3_runtime import create_environment


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    environment, source = create_environment(args)
    run_name = args.run_name or datetime.now().strftime(
        "stage3_moving_combat_%Y%m%d_%H%M%S"
    )
    model_directory = PROJECT_ROOT / "agent" / "models" / "stage_3" / run_name
    log_directory = PROJECT_ROOT / "agent" / "logs" / "stage_3" / run_name
    model_directory.mkdir(parents=True, exist_ok=False)
    log_directory.mkdir(parents=True, exist_ok=False)

    observation_shape = environment.observation_space.shape
    if observation_shape is None or len(observation_shape) != 1:
        raise ValueError("Stage Three requires a one-dimensional observation space")
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

    initialize_models(tf=tf, args=args, actor=actor, checkpoint=checkpoint)
    write_configuration(
        args=args,
        source=source,
        observation_size=observation_size,
        action_count=action_count,
        model_directory=model_directory,
    )

    summary_writer = tf.summary.create_file_writer(
        str(log_directory / "tensorboard")
    )
    episode_log = StageThreeEpisodeCsvLogger(log_directory / "episodes.csv")
    observation, _ = environment.reset(seed=args.seed)
    episode_state = StageThreeEpisodeState()
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
            rollout, observation, episode_state = collect_stage_three_rollout(
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
            write_rollout_summaries(
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
            print_rollout(step, args.timesteps, rollout, metrics)
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


if __name__ == "__main__":
    main()
