"""Stage Seven checkpoint initialization and Stage Six actor expansion."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from agent.training.stage_1.train_stage_1 import resolve_checkpoint
from agent.training.stage_2.stage_2_models import build_stage_two_actor


STAGE_SIX_OBSERVATION_SIZE = 30
STAGE_SEVEN_OBSERVATION_SIZE = 60
STAGE_SIX_ACTION_COUNT = 12
STAGE_SEVEN_ACTION_COUNT = 14


def initialize_models(*, tf: Any, args: argparse.Namespace, actor: Any, checkpoint: Any) -> None:
    if args.resume:
        path = resolve_checkpoint(tf, args.resume)
        _validate_stage_seven_checkpoint(tf, path)
        status = checkpoint.restore(path)
        status.expect_partial()
        status.assert_existing_objects_matched()
        print(f"Restored Stage Seven checkpoint: {path}")
    elif args.stage6_checkpoint:
        path = resolve_checkpoint(tf, args.stage6_checkpoint)
        transfer_stage_six_actor(
            tf=tf,
            stage_seven_actor=actor,
            checkpoint_path=path,
            hidden_size=args.hidden_size,
            new_action_bias=args.new_action_bias,
        )
        print(
            "Initialized Stage Seven actor from Stage Six; retained the first "
            "30 observations and 12 actions, added 30 neutral inputs and two "
            f"selection actions (bias {args.new_action_bias:+g}), and started "
            f"a fresh critic: {path}"
        )
    else:
        print("Initialized Stage Seven actor and critic from scratch.")


def transfer_stage_six_actor(
    *, tf: Any, stage_seven_actor: Any, checkpoint_path: str,
    hidden_size: int, new_action_bias: float,
) -> None:
    source = build_stage_two_actor(
        tf,
        observation_size=STAGE_SIX_OBSERVATION_SIZE,
        action_count=STAGE_SIX_ACTION_COUNT,
        hidden_size=hidden_size,
    )
    status = tf.train.Checkpoint(actor=source).restore(checkpoint_path)
    status.expect_partial()
    status.assert_existing_objects_matched()

    source_kernel, source_bias = source.layers[1].get_weights()
    target_kernel, _ = stage_seven_actor.layers[1].get_weights()
    if source_kernel.shape[0] != 30 or target_kernel.shape[0] != 60:
        raise ValueError("Unexpected Stage Six/Stage Seven observation shape")
    target_kernel.fill(0.0)
    target_kernel[:30] = source_kernel
    stage_seven_actor.layers[1].set_weights([target_kernel.astype(np.float32), source_bias])
    stage_seven_actor.layers[2].set_weights(source.layers[2].get_weights())

    source_policy_kernel, source_policy_bias = source.layers[3].get_weights()
    target_policy_kernel, target_policy_bias = stage_seven_actor.layers[3].get_weights()
    target_policy_kernel.fill(0.0)
    target_policy_bias.fill(float(new_action_bias))
    target_policy_kernel[:, :12] = source_policy_kernel
    target_policy_bias[:12] = source_policy_bias
    stage_seven_actor.layers[3].set_weights(
        [target_policy_kernel.astype(np.float32), target_policy_bias.astype(np.float32)]
    )


def _validate_stage_seven_checkpoint(tf: Any, checkpoint_path: str) -> None:
    names = {name for name, _ in tf.train.list_variables(checkpoint_path)}
    if not (any(name.startswith("actor_optimizer/") for name in names)
            and any(name.startswith("critic_optimizer/") for name in names)):
        raise ValueError("--resume requires a Stage Seven actor/critic checkpoint")
