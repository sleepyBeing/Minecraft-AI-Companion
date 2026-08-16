"""Stage Six checkpoint initialization and Stage Five actor expansion."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from agent.training.stage_1.train_stage_1 import resolve_checkpoint
from agent.training.stage_2.stage_2_models import build_stage_two_actor


STAGE_FIVE_OBSERVATION_SIZE = 22
STAGE_SIX_OBSERVATION_SIZE = 30
ACTION_COUNT = 12


def initialize_models(
    *,
    tf: Any,
    args: argparse.Namespace,
    actor: Any,
    checkpoint: Any,
) -> None:
    if args.resume:
        restored_path = resolve_checkpoint(tf, args.resume)
        _validate_stage_six_checkpoint(tf, restored_path)
        status = checkpoint.restore(restored_path)
        status.expect_partial()
        status.assert_existing_objects_matched()
        print(f"Restored Stage Six checkpoint: {restored_path}")
        return

    if args.stage5_checkpoint:
        restored_path = resolve_checkpoint(tf, args.stage5_checkpoint)
        transfer_stage_five_actor(
            tf=tf,
            stage_six_actor=actor,
            checkpoint_path=restored_path,
            hidden_size=args.hidden_size,
        )
        print(
            "Initialized Stage Six actor from Stage Five; copied 22 inputs "
            "and all 12 actions, initialized 8 protection inputs neutrally, "
            f"and started a fresh critic: {restored_path}"
        )
        return

    print("Initialized Stage Six actor and critic from scratch.")


def transfer_stage_five_actor(
    *,
    tf: Any,
    stage_six_actor: Any,
    checkpoint_path: str,
    hidden_size: int,
) -> None:
    source_actor = build_stage_two_actor(
        tf,
        observation_size=STAGE_FIVE_OBSERVATION_SIZE,
        action_count=ACTION_COUNT,
        hidden_size=hidden_size,
    )
    status = tf.train.Checkpoint(actor=source_actor).restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    source_hidden_one = source_actor.layers[1]
    target_hidden_one = stage_six_actor.layers[1]
    source_kernel, source_bias = source_hidden_one.get_weights()
    target_kernel, _ = target_hidden_one.get_weights()
    if source_kernel.shape[0] != STAGE_FIVE_OBSERVATION_SIZE or (
        target_kernel.shape[0] != STAGE_SIX_OBSERVATION_SIZE
    ):
        raise ValueError("Unexpected Stage Five/Stage Six observation shape")
    target_kernel.fill(0.0)
    target_kernel[:STAGE_FIVE_OBSERVATION_SIZE, :] = source_kernel
    target_hidden_one.set_weights(
        [target_kernel.astype(np.float32), source_bias]
    )
    stage_six_actor.layers[2].set_weights(source_actor.layers[2].get_weights())
    stage_six_actor.layers[3].set_weights(source_actor.layers[3].get_weights())


def _validate_stage_six_checkpoint(tf: Any, checkpoint_path: str) -> None:
    names = {name for name, _ in tf.train.list_variables(checkpoint_path)}
    if not (
        any(name.startswith("actor_optimizer/") for name in names)
        and any(name.startswith("critic_optimizer/") for name in names)
    ):
        raise ValueError(
            "--resume requires a Stage Six separate actor/critic checkpoint"
        )
