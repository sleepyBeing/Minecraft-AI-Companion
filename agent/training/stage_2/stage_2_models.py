"""Stage Two actor/critic construction and checkpoint-transfer helpers."""

from __future__ import annotations

from typing import Any

from agent.training.stage_1.train_stage_1 import build_actor_critic


def build_stage_two_actor(
    tf: Any,
    *,
    observation_size: int,
    action_count: int,
    hidden_size: int,
) -> Any:
    inputs = tf.keras.Input(
        shape=(observation_size,),
        dtype=tf.float32,
        name="actor_observation",
    )
    hidden = tf.keras.layers.Dense(
        hidden_size, activation="tanh", name="actor_hidden_1"
    )(inputs)
    hidden = tf.keras.layers.Dense(
        hidden_size, activation="tanh", name="actor_hidden_2"
    )(hidden)
    logits = tf.keras.layers.Dense(action_count, name="policy_logits")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="stage_two_actor")


def build_stage_two_critic(
    tf: Any,
    *,
    observation_size: int,
    hidden_size: int,
) -> Any:
    inputs = tf.keras.Input(
        shape=(observation_size,),
        dtype=tf.float32,
        name="critic_observation",
    )
    hidden = tf.keras.layers.Dense(
        hidden_size, activation="tanh", name="critic_hidden_1"
    )(inputs)
    hidden = tf.keras.layers.Dense(
        hidden_size, activation="tanh", name="critic_hidden_2"
    )(hidden)
    value = tf.keras.layers.Dense(1, name="critic_value")(hidden)
    return tf.keras.Model(inputs=inputs, outputs=value, name="stage_two_critic")


def transfer_stage_one_actor(
    *,
    tf: Any,
    stage_two_actor: Any,
    checkpoint_path: str,
    hidden_size: int,
) -> None:
    """Initialize only the Stage Two actor from Stage One positioning."""

    stage_one_model = build_actor_critic(
        tf,
        observation_size=10,
        action_count=7,
        hidden_size=hidden_size,
    )
    status = tf.train.Checkpoint(model=stage_one_model).restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    source_hidden_one = stage_one_model.layers[1]
    source_hidden_two = stage_one_model.layers[2]
    source_policy = stage_one_model.layers[3]
    target_hidden_one = stage_two_actor.layers[1]
    target_hidden_two = stage_two_actor.layers[2]
    target_policy = stage_two_actor.layers[3]

    source_kernel, source_bias = source_hidden_one.get_weights()
    target_kernel, _ = target_hidden_one.get_weights()
    target_kernel.fill(0.0)
    target_kernel[: source_kernel.shape[0], :] = source_kernel
    target_kernel[5, :] = 0.0
    target_hidden_one.set_weights([target_kernel, source_bias])
    target_hidden_two.set_weights(source_hidden_two.get_weights())

    source_policy_kernel, source_policy_bias = source_policy.get_weights()
    target_policy_kernel, target_policy_bias = target_policy.get_weights()
    target_policy_kernel.fill(0.0)
    target_policy_bias.fill(0.0)
    target_policy_kernel[:, : source_policy_kernel.shape[1]] = source_policy_kernel
    target_policy_bias[: source_policy_bias.shape[0]] = source_policy_bias
    target_policy_kernel[:, source_policy_kernel.shape[1]] = 0.0
    target_policy_bias[source_policy_bias.shape[0]] = -1.5
    target_policy.set_weights([target_policy_kernel, target_policy_bias])


def transfer_legacy_stage_two_actor(
    *,
    tf: Any,
    stage_two_actor: Any,
    checkpoint_path: str,
    hidden_size: int,
    observation_size: int,
    action_count: int,
) -> None:
    """Copy the actor portion of a pre-separation Stage Two checkpoint."""

    legacy_model = build_actor_critic(
        tf,
        observation_size=observation_size,
        action_count=action_count,
        hidden_size=hidden_size,
    )
    status = tf.train.Checkpoint(model=legacy_model).restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    for source_layer, target_layer in zip(
        legacy_model.layers[1:4],
        stage_two_actor.layers[1:4],
        strict=True,
    ):
        target_layer.set_weights(source_layer.get_weights())
