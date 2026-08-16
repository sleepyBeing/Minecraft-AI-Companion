"""Command-line configuration for Stage Six PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.training.stage_1.train_stage_1 import (
    positive_float,
    positive_integer,
    unit_interval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMESTEPS = 250_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TensorFlow PPO to protect the Stage Six NPC."
    )
    parser.add_argument(
        "--timesteps", type=positive_integer, default=DEFAULT_TIMESTEPS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Use the local Stage Six simulation instead of Minecraft.",
    )
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--bridge-timeout", type=positive_float, default=20.0)
    parser.add_argument("--rollout-steps", type=positive_integer, default=2_048)
    parser.add_argument("--batch-size", type=positive_integer, default=64)

    actor = parser.add_argument_group("actor")
    actor.add_argument("--epochs", type=positive_integer, default=10)
    actor.add_argument("--learning-rate", type=positive_float, default=3e-4)
    actor.add_argument("--clip-ratio", type=positive_float, default=0.2)
    actor.add_argument("--target-kl", type=positive_float, default=0.03)
    actor.add_argument("--entropy-coefficient", type=float, default=0.05)
    actor.add_argument("--max-gradient-norm", type=positive_float, default=0.5)
    actor.add_argument("--hidden-size", type=positive_integer, default=128)

    critic = parser.add_argument_group("critic")
    critic.add_argument("--critic-epochs", type=positive_integer, default=12)
    critic.add_argument(
        "--critic-learning-rate", type=positive_float, default=1.5e-4
    )
    critic.add_argument(
        "--critic-hidden-size", type=positive_integer, default=128
    )
    critic.add_argument(
        "--critic-loss", choices=("huber", "mse"), default="huber"
    )
    critic.add_argument(
        "--critic-huber-delta", type=positive_float, default=5.0
    )
    critic.add_argument(
        "--critic-max-gradient-norm", type=positive_float, default=1.0
    )
    critic.add_argument("--value-coefficient", type=positive_float, default=1.0)

    returns = parser.add_argument_group("returns")
    returns.add_argument("--gamma", type=unit_interval, default=0.99)
    returns.add_argument("--gae-lambda", type=unit_interval, default=0.95)

    parser.add_argument(
        "--checkpoint-freq", type=positive_integer, default=10_000
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--resume",
        type=Path,
        help="Stage Six actor/critic checkpoint to restore.",
    )
    initialization.add_argument(
        "--stage5-checkpoint",
        type=Path,
        help=(
            "Initialize the expanded actor from Stage Five; the Stage Six "
            "critic starts fresh."
        ),
    )
    initialization.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize both Stage Six networks from scratch.",
    )
    parser.add_argument("--run-name")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size > args.rollout_steps:
        raise ValueError("--batch-size cannot exceed --rollout-steps")
