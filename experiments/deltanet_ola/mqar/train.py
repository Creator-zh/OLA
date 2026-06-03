from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments.deltanet_ola.env import collect_run_metadata
from experiments.deltanet_ola.methods import build_mixer
from experiments.deltanet_ola.metrics import peak_memory_bytes, reset_peak_memory, throughput_metrics
from experiments.deltanet_ola.mqar.data import MQARConfig, generate_mqar_batch, query_accuracy, sequence_loss
from experiments.deltanet_ola.mqar.sweep import MQARSweepConfig, build_mqar_jobs
from experiments.ola_mqar.results import append_jsonl, write_csv


class MQARModel(nn.Module):
    def __init__(
        self,
        method: str,
        vocabulary_size: int,
        model_hidden_size: int,
        state_matrix_dimension: int,
        delta_num_heads: int,
        delta_use_short_conv: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, model_hidden_size)
        self.mixer = build_mixer(
            method=method,
            hidden_size=model_hidden_size,
            state_matrix_dimension=state_matrix_dimension,
            delta_num_heads=delta_num_heads,
            delta_use_short_conv=delta_use_short_conv,
        )
        self.norm = nn.LayerNorm(model_hidden_size)
        self.output_projection = nn.Linear(model_hidden_size, vocabulary_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embedding(input_ids)
        mixed_hidden, aux = self.mixer(hidden)
        logits = self.output_projection(self.norm(mixed_hidden))
        return logits, aux


@torch.no_grad()
def evaluate(
    model: MQARModel,
    config: MQARConfig,
    batch_size: int,
    evaluation_batch_count: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_queries = 0
    for _ in range(evaluation_batch_count):
        batch = generate_mqar_batch(config, batch_size, device, generator)
        logits, _ = model(batch.input_ids)
        query_count = int(batch.query_mask.sum().item())
        total_loss += sequence_loss(logits, batch.labels).item() * query_count
        total_correct += ((logits.argmax(dim=-1) == batch.labels) & batch.query_mask).sum().item()
        total_queries += query_count
    return total_loss / max(total_queries, 1), total_correct / max(total_queries, 1)


def run_job(job: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(job["random_seed"])
    data_config = MQARConfig(
        vocabulary_size=job["vocabulary_size"],
        sequence_length=job["sequence_length"],
        number_of_key_value_pairs=job["number_of_key_value_pairs"],
        power_a=job["power_a"],
        random_non_queries=job["random_non_queries"],
    )
    model = MQARModel(
        method=job["method"],
        vocabulary_size=job["vocabulary_size"],
        model_hidden_size=job["model_hidden_size"],
        state_matrix_dimension=job["state_matrix_dimension"],
        delta_num_heads=job["delta_num_heads"],
        delta_use_short_conv=job["delta_use_short_conv"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["learning_rate"], weight_decay=0.01)
    train_generator = torch.Generator(device=device).manual_seed(job["random_seed"] + 100_000)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    reset_peak_memory(device)

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(job["training_steps"]):
        model.train()
        batch = generate_mqar_batch(data_config, job["training_batch_size"], device, train_generator)
        logits, _ = model(batch.input_ids)
        loss = sequence_loss(logits, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())

    elapsed = time.perf_counter() - start
    eval_loss, eval_accuracy = evaluate(
        model=model,
        config=data_config,
        batch_size=job["training_batch_size"],
        evaluation_batch_count=job["evaluation_batch_count"],
        device=device,
        seed=job["random_seed"] + 200_000,
    )
    tokens = job["training_steps"] * job["training_batch_size"] * job["sequence_length"]
    return {
        **job,
        **collect_run_metadata(device=device, dtype=str(next(model.parameters()).dtype)),
        "parameter_count": parameter_count,
        "train_loss": last_loss,
        "final_evaluation_loss": eval_loss,
        "final_evaluation_accuracy": eval_accuracy,
        "peak_memory_bytes": peak_memory_bytes(device),
        "elapsed_seconds": elapsed,
        **throughput_metrics(tokens=tokens, steps=job["training_steps"], elapsed_seconds=elapsed),
    }


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_method_tuple(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(methods) - {"fla_delta", "ola"})
    if invalid:
        raise ValueError(f"unsupported methods: {invalid}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeltaNet-paper MQAR reproduction with OLA comparison.")
    parser.add_argument("--methods", type=_parse_method_tuple, default=("fla_delta", "ola"))
    parser.add_argument("--vocabulary-size", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--number-of-key-value-pairs", type=int, default=64)
    parser.add_argument("--model-hidden-sizes", type=_parse_int_tuple, default=(64, 128, 256, 512))
    parser.add_argument("--random-seeds", type=_parse_int_tuple, default=(1, 2, 3))
    parser.add_argument("--training-steps", type=int, default=10_000)
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-count", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--state-matrix-dimension", type=int, default=None)
    parser.add_argument("--delta-num-heads", type=int, default=2)
    parser.add_argument("--delta-use-short-conv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--power-a", type=float, default=0.01)
    parser.add_argument("--random-non-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-jsonl", type=Path, default=Path("runs/deltanet_ola/mqar.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/deltanet_ola/mqar.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = MQARSweepConfig(
        methods=args.methods,
        vocabulary_size=args.vocabulary_size,
        sequence_length=args.sequence_length,
        number_of_key_value_pairs=args.number_of_key_value_pairs,
        model_hidden_sizes=args.model_hidden_sizes,
        random_seeds=args.random_seeds,
        training_steps=args.training_steps,
        training_batch_size=args.training_batch_size,
        evaluation_batch_count=args.evaluation_batch_count,
        learning_rate=args.learning_rate,
        state_matrix_dimension=args.state_matrix_dimension,
        delta_num_heads=args.delta_num_heads,
        delta_use_short_conv=args.delta_use_short_conv,
        power_a=args.power_a,
        random_non_queries=args.random_non_queries,
    )
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(build_mqar_jobs(config), start=1):
        print(
            f"job={index} method={job['method']} hidden={job['model_hidden_size']} seed={job['random_seed']}",
            flush=True,
        )
        row = run_job(job, device)
        rows.append(row)
        append_jsonl(args.out_jsonl, row)
        write_csv(args.out_csv, rows)
        print(
            f"method={row['method']} hidden={row['model_hidden_size']} seed={row['random_seed']} "
            f"loss={row['final_evaluation_loss']:.4f} acc={row['final_evaluation_accuracy']:.4f} "
            f"tokens_per_second={row['tokens_per_second']:.2f} peak_memory_bytes={row['peak_memory_bytes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

