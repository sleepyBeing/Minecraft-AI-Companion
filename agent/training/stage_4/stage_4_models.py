"""Stage Four checkpoint initialization and Stage Three actor expansion."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from agent.training.stage_1.train_stage_1 import resolve_checkpoint
from agent.training.stage_2.stage_2_models import build_stage_two_actor


STAGE_THREE_OBSERVATION_SIZE = 13
STAGE_THREE_ACTION_COUNT = 8


def initialize_models(
    *,
    tf: Any,
    args: argparse.Namespace,
    actor: Any,
    checkpoint: Any,
) -> None:
    if args.resume:
        restored_path = resolve_checkpoint(tf, args.resume)
        names = {name for name, _ in tf.train.list_variables(restored_path)}
        if not (
            any(name.startswith("actor_optimizer/") for name in names)
            and any(name.startswith("critic_optimizer/") for name in names)
        ):
            raise ValueError(
                "--resume requires a Stage Four separate actor/critic checkpoint"
            )
        status = checkpoint.restore(restored_path)
        status.expect_partial()
        status.assert_existing_objects_matched()
        print(f"Restored Stage Four checkpoint: {restored_path}")
        return

    if args.stage3_checkpoint:
        restored_path = resolve_checkpoint(tf, args.stage3_checkpoint)
        transfer_stage_three_actor(
            tf=tf,
            stage_four_actor=actor,
            checkpoint_path=restored_path,
            hidden_size=args.hidden_size,
            new_action_bias=args.new_action_bias,
        )
        print(
            "Initialized Stage Four actor from Stage Three; copied the "
            "original 13 inputs and 8 actions, initialized 5 new inputs and "
            f"2 new actions, and started a fresh critic: {restored_path}"
        )
        return

    print("Initialized Stage Four actor and critic from scratch.")


def transfer_stage_three_actor(
    *,
    tf: Any,
    stage_four_actor: Any,
    checkpoint_path: str,
    hidden_size: int,
    new_action_bias: float,
) -> None:
    source_actor = build_stage_two_actor(
        tf,
        observation_size=STAGE_THREE_OBSERVATION_SIZE,
        action_count=STAGE_THREE_ACTION_COUNT,
        hidden_size=hidden_size,
    )
    status = tf.train.Checkpoint(actor=source_actor).restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    source_hidden_one = source_actor.layers[1]
    source_hidden_two = source_actor.layers[2]
    source_policy = source_actor.layers[3]
    target_hidden_one = stage_four_actor.layers[1]
    target_hidden_two = stage_four_actor.layers[2]
    target_policy = stage_four_actor.layers[3]

    source_kernel, source_bias = source_hidden_one.get_weights()
    target_kernel, _ = target_hidden_one.get_weights()
    if target_kernel.shape[0] < source_kernel.shape[0]:
        raise ValueError("Stage Four actor has fewer observation inputs than Stage Three")
    target_kernel.fill(0.0)
    target_kernel[: source_kernel.shape[0], :] = source_kernel
    target_hidden_one.set_weights([target_kernel, source_bias])
    target_hidden_two.set_weights(source_hidden_two.get_weights())

    source_policy_kernel, source_policy_bias = source_policy.get_weights()
    target_policy_kernel, target_policy_bias = target_policy.get_weights()
    if target_policy_kernel.shape[1] < source_policy_kernel.shape[1]:
        raise ValueError("Stage Four actor has fewer actions than Stage Three")
    target_policy_kernel.fill(0.0)
    target_policy_bias.fill(float(new_action_bias))
    target_policy_kernel[:, : source_policy_kernel.shape[1]] = source_policy_kernel
    target_policy_bias[: source_policy_bias.shape[0]] = source_policy_bias
    target_policy.set_weights(
        [
            target_policy_kernel.astype(np.float32),
            target_policy_bias.astype(np.float32),
        ]
    )
