"""Independent actor and critic updates for Stage Two PPO."""

from __future__ import annotations

from typing import Any

import numpy as np


def update_stage_two_ppo(
    *,
    tf: Any,
    actor: Any,
    critic: Any,
    actor_optimizer: Any,
    critic_optimizer: Any,
    rollout: dict[str, np.ndarray],
    actor_epochs: int,
    critic_epochs: int,
    batch_size: int,
    clip_ratio: float,
    entropy_coefficient: float,
    value_coefficient: float,
    actor_max_gradient_norm: float,
    critic_max_gradient_norm: float,
    critic_loss_name: str,
    critic_huber_delta: float,
) -> dict[str, float]:

    sample_count = len(rollout["actions"])
    actor_metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "entropy": [],
        "approximate_kl": [],
        "clip_fraction": [],
        "actor_gradient_norm": [],
    }
    critic_losses: list[float] = []
    critic_gradient_norms: list[float] = []

    for _ in range(actor_epochs):
        shuffled_indices = np.random.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            indices = shuffled_indices[start : start + batch_size]
            observations = tf.convert_to_tensor(rollout["observations"][indices])
            actions = tf.convert_to_tensor(rollout["actions"][indices])
            old_log_probabilities = tf.convert_to_tensor(
                rollout["log_probabilities"][indices]
            )
            advantages = tf.convert_to_tensor(rollout["advantages"][indices])

            with tf.GradientTape() as tape:
                logits = actor(observations, training=True)
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
                    tf.clip_by_value(
                        ratios,
                        1.0 - clip_ratio,
                        1.0 + clip_ratio,
                    )
                    * advantages
                )
                policy_loss = -tf.reduce_mean(
                    tf.minimum(unclipped_objective, clipped_objective)
                )
                probabilities = tf.nn.softmax(logits)
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(
                        probabilities * all_log_probabilities,
                        axis=1,
                    )
                )
                actor_loss = policy_loss - entropy_coefficient * entropy

            gradients = tape.gradient(actor_loss, actor.trainable_variables)
            gradients, gradient_norm = tf.clip_by_global_norm(
                gradients, actor_max_gradient_norm
            )
            actor_optimizer.apply_gradients(
                zip(gradients, actor.trainable_variables)
            )

            approximate_kl = tf.reduce_mean(
                old_log_probabilities - new_log_probabilities
            )
            clip_fraction = tf.reduce_mean(
                tf.cast(
                    tf.abs(ratios - 1.0) > clip_ratio,
                    tf.float32,
                )
            )
            for name, value in {
                "policy_loss": policy_loss,
                "entropy": entropy,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
                "actor_gradient_norm": gradient_norm,
            }.items():
                actor_metrics[name].append(float(value.numpy()))

    for _ in range(critic_epochs):
        shuffled_indices = np.random.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            indices = shuffled_indices[start : start + batch_size]
            observations = tf.convert_to_tensor(rollout["observations"][indices])
            returns = tf.convert_to_tensor(rollout["returns"][indices])

            with tf.GradientTape() as tape:
                predicted_values = tf.squeeze(
                    critic(observations, training=True), axis=1
                )
                errors = predicted_values - returns
                if critic_loss_name == "huber":
                    absolute_errors = tf.abs(errors)
                    quadratic = tf.minimum(
                        absolute_errors, critic_huber_delta
                    )
                    linear = absolute_errors - quadratic
                    value_loss = tf.reduce_mean(
                        0.5 * tf.square(quadratic)
                        + critic_huber_delta * linear
                    )
                elif critic_loss_name == "mse":
                    value_loss = 0.5 * tf.reduce_mean(tf.square(errors))
                else:
                    raise ValueError(
                        f"Unsupported critic loss: {critic_loss_name}"
                    )
                critic_objective = value_coefficient * value_loss

            gradients = tape.gradient(
                critic_objective, critic.trainable_variables
            )
            gradients, gradient_norm = tf.clip_by_global_norm(
                gradients, critic_max_gradient_norm
            )
            critic_optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables)
            )
            critic_losses.append(float(value_loss.numpy()))
            critic_gradient_norms.append(float(gradient_norm.numpy()))

    return {
        **{
            name: float(np.mean(values))
            for name, values in actor_metrics.items()
        },
        "value_loss": float(np.mean(critic_losses)),
        "critic_gradient_norm": float(np.mean(critic_gradient_norms)),
    }
