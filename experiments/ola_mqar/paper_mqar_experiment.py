from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments.ola_mqar.layers import OLAMixer
from experiments.ola_mqar.multiquery_experiment import (
    MultiQueryConfig,
    generate_multiquery_batch,
    query_accuracy,
    sequence_loss,
)
from experiments.ola_mqar.results import append_jsonl, write_csv


@dataclass(frozen=True)
class PaperMQARConfig:
    methods: tuple[str, ...] = ("fla_delta", "ola")
    vocabulary_size: int = 8192
    sequence_lengths: tuple[int, ...] = (512,)
    number_of_key_value_pairs: tuple[int, ...] = (64,)
    model_hidden_sizes: tuple[int, ...] = (64, 128, 256, 512)
    random_seeds: tuple[int, ...] = (1, 2, 3)
    training_steps: int = 10_000
    training_batch_size: int = 256
    evaluation_batch_count: int = 32
    learning_rate: float = 0.003
    state_matrix_dimension: int | None = None
    delta_num_heads: int = 2
    delta_use_short_conv: bool = False
    power_a: float = 0.01
    random_non_queries: bool = True


class PaperMQARModel(nn.Module):
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
        if method not in {"fla_delta", "ola"}:
            raise ValueError(f"method must be 'fla_delta' or 'ola', got {method!r}")
        self.method = method
        self.embedding = nn.Embedding(vocabulary_size, model_hidden_size)
        if method == "fla_delta":
            fla_layers = importlib.import_module("fla.layers")
            self.mixer = fla_layers.DeltaNet(
                mode="chunk",
                hidden_size=model_hidden_size,
                expand_k=state_matrix_dimension / model_hidden_size,
                expand_v=state_matrix_dimension / model_hidden_size,
                num_heads=delta_num_heads,
                use_beta=True,
                use_gate=False,
                use_short_conv=delta_use_short_conv,
                qk_activation="silu",
                qk_norm="l2",
            )
        else:
            self.mixer = OLAMixer(model_hidden_size, state_matrix_dimension)
        self.norm = nn.LayerNorm(model_hidden_size)
        self.output_projection = nn.Linear(model_hidden_size, vocabulary_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embedding(input_ids)
        if self.method == "fla_delta":
            mixed_hidden, _, _ = self.mixer(hidden)
            aux: dict[str, torch.Tensor] = {}
        else:
            mixed = self.mixer(hidden)
            mixed_hidden = mixed.hidden_states
            aux = mixed.aux
        logits = self.output_projection(self.norm(mixed_hidden))
        return logits, aux


def build_paper_jobs(config: PaperMQARConfig) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for method in config.methods:
        for sequence_length in config.sequence_lengths:
            for number_of_key_value_pairs in config.number_of_key_value_pairs:
                for model_hidden_size in config.model_hidden_sizes:
                    state_matrix_dimension = config.state_matrix_dimension or model_hidden_size
                    for random_seed in config.random_seeds:
                        jobs.append(
                            {
                                "method": method,
                                "vocabulary_size": config.vocabulary_size,
                                "sequence_length": sequence_length,
                                "number_of_key_value_pairs": number_of_key_value_pairs,
                                "model_hidden_size": model_hidden_size,
                                "state_matrix_dimension": state_matrix_dimension,
                                "random_seed": random_seed,
                                "training_steps": config.training_steps,
                                "training_batch_size": config.training_batch_size,
                                "evaluation_batch_count": config.evaluation_batch_count,
                                "learning_rate": config.learning_rate,
                                "delta_num_heads": config.delta_num_heads,
                                "delta_use_short_conv": config.delta_use_short_conv,
                                "power_a": config.power_a,
                                "random_non_queries": config.random_non_queries,
                            }
                        )
    return jobs


@torch.no_grad()
def evaluate(
    model: PaperMQARModel,
    data_config: MultiQueryConfig,
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
        batch = generate_multiquery_batch(data_config, batch_size, device, generator)
        logits, _ = model(batch.input_ids)
        query_count = batch.query_mask.sum().item()
        total_loss += sequence_loss(logits, batch.labels).item() * query_count
        total_correct += ((logits.argmax(dim=-1) == batch.labels) & batch.query_mask).sum().item()
        total_queries += query_count
    return total_loss / total_queries, total_correct / total_queries


def run_job(job: dict[str, Any], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(job["random_seed"])
    data_config = MultiQueryConfig(
        vocabulary_size=job["vocabulary_size"],
        sequence_length=job["sequence_length"],
        number_of_key_value_pairs=job["number_of_key_value_pairs"],
        power_a=job["power_a"],
        random_non_queries=job["random_non_queries"],
    )
    model = PaperMQARModel(
        method=job["method"],
        vocabulary_size=job["vocabulary_size"],
        model_hidden_size=job["model_hidden_size"],
        state_matrix_dimension=job["state_matrix_dimension"],
        delta_num_heads=job["delta_num_heads"],
        delta_use_short_conv=job["delta_use_short_conv"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["learning_rate"], weight_decay=0.01)
    train_generator = torch.Generator(device=device).manual_seed(job["random_seed"] + 100_000)

    start = time.perf_counter()
    for _ in range(job["training_steps"]):
        model.train()
        batch = generate_multiquery_batch(data_config, job["training_batch_size"], device, train_generator)
        logits, _ = model(batch.input_ids)
        loss = sequence_loss(logits, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_evaluation_loss, final_evaluation_accuracy = evaluate(
        model=model,
        data_config=data_config,
        batch_size=job["training_batch_size"],
        evaluation_batch_count=job["evaluation_batch_count"],
        device=device,
        seed=job["random_seed"] + 200_000,
    )
    return {
        **job,
        "final_evaluation_loss": final_evaluation_loss,
        "final_evaluation_accuracy": final_evaluation_accuracy,
        "steps_per_second": job["training_steps"] / max(time.perf_counter() - start, 1e-9),
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
    parser = argparse.ArgumentParser(description="Run paper-style MQAR comparison: FLA DeltaNet vs OLA.")
    parser.add_argument("--methods", type=_parse_method_tuple, default=("fla_delta", "ola"))
    parser.add_argument("--vocabulary-size", type=int, default=8192)
    parser.add_argument("--sequence-lengths", type=_parse_int_tuple, default=(512,))
    parser.add_argument("--number-of-key-value-pairs", type=_parse_int_tuple, default=(64,))
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-jsonl", type=Path, default=Path("runs/ola_mqar/paper_mqar.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/ola_mqar/paper_mqar.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = PaperMQARConfig(
        methods=args.methods,
        vocabulary_size=args.vocabulary_size,
        sequence_lengths=args.sequence_lengths,
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
    jobs = build_paper_jobs(config)
    print(f"running_jobs={len(jobs)} device={device}", flush=True)
    for index, job in enumerate(jobs, start=1):
        print(
            f"job={index}/{len(jobs)} method={job['method']} "
            f"sequence_length={job['sequence_length']} number_of_key_value_pairs={job['number_of_key_value_pairs']} "
            f"model_hidden_size={job['model_hidden_size']} random_seed={job['random_seed']}",
            flush=True,
        )
        row = run_job(job, device)
        rows.append(row)
        append_jsonl(args.out_jsonl, row)
        write_csv(args.out_csv, rows)
        print(
            f"result method={row['method']} model_hidden_size={row['model_hidden_size']} "
            f"random_seed={row['random_seed']} final_evaluation_loss={row['final_evaluation_loss']:.4f} "
            f"final_evaluation_accuracy={row['final_evaluation_accuracy']:.4f} "
            f"steps_per_second={row['steps_per_second']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
