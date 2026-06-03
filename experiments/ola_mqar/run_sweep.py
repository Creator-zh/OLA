from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.ola_mqar.results import append_jsonl, write_csv
from experiments.ola_mqar.train import train_method


@dataclass(frozen=True)
class SweepConfig:
    methods: tuple[str, ...]
    num_pairs: tuple[int, ...]
    seeds: tuple[int, ...]
    steps: int
    eval_interval: int
    batch_size: int
    eval_batches: int
    vocab_size: int
    input_seq_len: int
    power_a: float
    d_models: tuple[int, ...]
    state_dim: int
    lr: float
    device: str


def _parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def _parse_method_list(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(parsed) - {"delta", "ola"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported methods: {', '.join(invalid)}")
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one method.")
    return parsed


def build_jobs(config: SweepConfig) -> list[dict[str, Any]]:
    jobs = []
    for method in config.methods:
        for num_pairs in config.num_pairs:
            for d_model in config.d_models:
                for seed in config.seeds:
                    jobs.append(
                        {
                            "method": method,
                            "num_pairs": num_pairs,
                            "seed": seed,
                            "steps": config.steps,
                            "eval_interval": config.eval_interval,
                            "batch_size": config.batch_size,
                            "eval_batches": config.eval_batches,
                            "vocab_size": config.vocab_size,
                            "input_seq_len": config.input_seq_len,
                            "d_model": d_model,
                            "state_dim": config.state_dim,
                            "power_a": config.power_a,
                            "lr": config.lr,
                            "device": config.device,
                            "model_initialization_seed": seed,
                            "training_data_seed": seed + 100_000,
                            "evaluation_data_seed": seed + 200_000,
                        }
                    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable Delta vs OLA sweeps on MQAR.")
    parser.add_argument("--methods", type=_parse_method_list, default=("delta", "ola"))
    parser.add_argument("--num-pairs-list", type=_parse_int_list, default=(2, 4, 8))
    parser.add_argument("--seeds", type=_parse_int_list, default=(1, 2, 3))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--input-seq-len", type=int, default=64)
    parser.add_argument("--power-a", type=float, default=0.01)
    parser.add_argument("--d-model-list", type=_parse_int_list, default=(128,))
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-jsonl", type=Path, default=Path("runs/ola_mqar/sweep.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/ola_mqar/sweep.csv"))
    parser.add_argument("--quiet-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SweepConfig(
        methods=args.methods,
        num_pairs=args.num_pairs_list,
        seeds=args.seeds,
        steps=args.steps,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        input_seq_len=args.input_seq_len,
        power_a=args.power_a,
        d_models=args.d_model_list,
        state_dim=args.state_dim,
        lr=args.lr,
        device=args.device,
    )
    jobs = build_jobs(config)
    rows: list[dict[str, Any]] = []
    device = torch.device(config.device)

    print(f"running_jobs={len(jobs)} out_jsonl={args.out_jsonl} out_csv={args.out_csv}", flush=True)
    for index, job in enumerate(jobs, start=1):
        print(
            f"job={index}/{len(jobs)} method={job['method']} "
            f"num_pairs={job['num_pairs']} d_model={job['d_model']} seed={job['seed']}",
            flush=True,
        )
        _, metrics = train_method(
            method=job["method"],
            steps=job["steps"],
            eval_interval=job["eval_interval"],
            batch_size=job["batch_size"],
            eval_batches=job["eval_batches"],
            vocab_size=job["vocab_size"],
            input_seq_len=job["input_seq_len"],
            num_pairs=job["num_pairs"],
            power_a=job["power_a"],
            d_model=job["d_model"],
            state_dim=job["state_dim"],
            lr=job["lr"],
            seed=job["model_initialization_seed"],
            device=device,
            train_data_seed=job["training_data_seed"],
            eval_data_seed=job["evaluation_data_seed"],
            log=not args.quiet_train,
        )
        row = {**job, **metrics}
        rows.append(row)
        append_jsonl(args.out_jsonl, row)
        write_csv(args.out_csv, rows)
        print(
            f"result method={row['method']} num_pairs={row['num_pairs']} seed={row['seed']} "
            f"d_model={row['d_model']} "
            f"eval_loss={row['eval_loss']:.4f} eval_acc={row['eval_acc']:.4f} "
            f"steps_per_sec={row['steps_per_sec']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
