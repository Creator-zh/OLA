from __future__ import annotations

import argparse

import torch

from experiments.ola_mqar.train import train_method


def format_comparison(delta: dict[str, float], ola: dict[str, float]) -> str:
    return (
        "comparison "
        f"delta_loss={delta['eval_loss']:.4f} delta_acc={delta['eval_acc']:.4f} "
        f"ola_loss={ola['eval_loss']:.4f} ola_acc={ola['eval_acc']:.4f} "
        f"ola_minus_delta_loss={ola['eval_loss'] - delta['eval_loss']:.4f} "
        f"ola_minus_delta_acc={ola['eval_acc'] - delta['eval_acc']:.4f} "
        f"delta_steps_per_sec={delta['steps_per_sec']:.2f} "
        f"ola_steps_per_sec={ola['steps_per_sec']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paired Delta vs OLA MQAR comparison.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-data-seed", type=int, default=None)
    parser.add_argument("--eval-data-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _, delta_metrics = train_method(
        method="delta",
        steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        d_model=args.d_model,
        state_dim=args.state_dim,
        lr=args.lr,
        seed=args.seed,
        device=device,
        train_data_seed=args.train_data_seed,
        eval_data_seed=args.eval_data_seed,
        log=True,
    )
    _, ola_metrics = train_method(
        method="ola",
        steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        num_pairs=args.num_pairs,
        d_model=args.d_model,
        state_dim=args.state_dim,
        lr=args.lr,
        seed=args.seed,
        device=device,
        train_data_seed=args.train_data_seed,
        eval_data_seed=args.eval_data_seed,
        log=True,
    )
    print(format_comparison(delta_metrics, ola_metrics), flush=True)


if __name__ == "__main__":
    main()
