"""Stage Three checkpoint initialization and Stage Two actor transfer."""

from __future__ import annotations

import argparse
from typing import Any

from agent.training.stage_1.train_stage_1 import resolve_checkpoint


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
                "--resume requires a Stage Three separate actor/critic checkpoint"
            )
        checkpoint.restore(restored_path).expect_partial()
        print(f"Restored Stage Three checkpoint: {restored_path}")
    elif args.stage2_checkpoint:
        restored_path = resolve_checkpoint(tf, args.stage2_checkpoint)
        transfer_checkpoint = tf.train.Checkpoint(actor=actor)
        status = transfer_checkpoint.restore(restored_path)
        status.expect_partial()
        status.assert_existing_objects_matched()
        print(
            "Initialized Stage Three actor from Stage Two; "
            f"critic started fresh: {restored_path}"
        )
