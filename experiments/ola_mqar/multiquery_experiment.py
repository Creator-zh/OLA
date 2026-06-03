from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ola_mqar.layers import DeltaMixer, OLAMixer
from experiments.ola_mqar.results import append_jsonl, write_csv


@dataclass(frozen=True)
class MultiQueryConfig:
    vocabulary_size: int = 64
    sequence_length: int = 16
    number_of_key_value_pairs: int = 2
    power_a: float = 0.01
    random_non_queries: bool = True


@dataclass(frozen=True)
class MultiQueryBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_mask: torch.Tensor


class MultiQueryModel(nn.Module):
    def __init__(
        self,
        method: str,
        vocabulary_size: int,
        model_hidden_size: int,
        state_matrix_dimension: int,
    ):
        super().__init__()
        if method not in {"delta", "ola"}:
            raise ValueError(f"method must be 'delta' or 'ola', got {method!r}")
        self.embedding = nn.Embedding(vocabulary_size, model_hidden_size)
        self.mixer = (
            DeltaMixer(model_hidden_size, state_matrix_dimension)
            if method == "delta"
            else OLAMixer(model_hidden_size, state_matrix_dimension)
        )
        self.norm = nn.LayerNorm(model_hidden_size)
        self.output_projection = nn.Linear(model_hidden_size, vocabulary_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embedding(input_ids)
        mixed = self.mixer(hidden)
        logits = self.output_projection(self.norm(mixed.hidden_states))
        return logits, mixed.aux


def _validate_config(config: MultiQueryConfig) -> None:
    if config.sequence_length % 2 != 0:
        raise ValueError("sequence_length must be even.")
    if config.number_of_key_value_pairs * 4 > config.sequence_length:
        raise ValueError("sequence_length must fit key-value context and query slots.")
    if config.vocabulary_size <= config.sequence_length:
        raise ValueError("vocabulary_size must be larger than sequence_length.")
    if config.number_of_key_value_pairs >= config.vocabulary_size // 2:
        raise ValueError("number_of_key_value_pairs must fit token ranges.")
    if config.power_a <= 0:
        raise ValueError("power_a must be positive.")


def generate_multiquery_batch(
    config: MultiQueryConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> MultiQueryBatch:
    _validate_config(config)
    # Match Zoology MQAR: build one extra position, then shift labels left so the
    # target value is scored at the query token position.
    examples = torch.zeros(batch_size, config.sequence_length + 1, dtype=torch.long, device=device)
    raw_labels = torch.full((batch_size, config.sequence_length + 1), -100, dtype=torch.long, device=device)
    labels = torch.full((batch_size, config.sequence_length), -100, dtype=torch.long, device=device)
    key_pool = torch.arange(1, config.vocabulary_size // 2, device=device)
    value_pool = torch.arange(config.vocabulary_size // 2, config.vocabulary_size, device=device)
    context_length = config.number_of_key_value_pairs * 2
    gap_space = (config.sequence_length - context_length) // 2
    gap_weights = config.power_a * torch.arange(
        1,
        gap_space + 1,
        device=device,
        dtype=torch.float32,
    ).pow(config.power_a - 1)
    gap_weights = gap_weights / gap_weights.sum()

    for batch_index in range(batch_size):
        keys = key_pool[
            torch.randperm(key_pool.numel(), device=device, generator=generator)[: config.number_of_key_value_pairs]
        ]
        values = value_pool[
            torch.randperm(value_pool.numel(), device=device, generator=generator)[: config.number_of_key_value_pairs]
        ]
        gaps = torch.multinomial(gap_weights, config.number_of_key_value_pairs, replacement=False, generator=generator)

        examples[batch_index, 0:context_length:2] = keys
        examples[batch_index, 1:context_length:2] = values
        query_positions = context_length + gaps * 2
        examples[batch_index, query_positions] = keys
        raw_labels[batch_index, query_positions + 1] = values

    input_ids = examples[:, :-1]
    labels = raw_labels[:, 1:]
    if config.random_non_queries:
        zero_mask = input_ids == 0
        random_tokens = torch.randint(
            config.vocabulary_size,
            size=input_ids.shape,
            device=device,
            generator=generator,
        )
        input_ids[zero_mask] = random_tokens[zero_mask]
    return MultiQueryBatch(input_ids=input_ids, labels=labels, query_mask=labels != -100)


def sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def query_accuracy(logits: torch.Tensor, labels: torch.Tensor, query_mask: torch.Tensor) -> float:
    correct = ((logits.argmax(dim=-1) == labels) & query_mask).sum().item()
    total = query_mask.sum().item()
    return correct / total


@torch.no_grad()
def evaluate(
    model: MultiQueryModel,
    config: MultiQueryConfig,
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
        batch = generate_multiquery_batch(config, batch_size, device, generator)
        logits, _ = model(batch.input_ids)
        total_loss += sequence_loss(logits, batch.labels).item() * batch.query_mask.sum().item()
        total_correct += ((logits.argmax(dim=-1) == batch.labels) & batch.query_mask).sum().item()
        total_queries += batch.query_mask.sum().item()
    return total_loss / total_queries, total_correct / total_queries


def train_once(
    method: str,
    config: MultiQueryConfig,
    model_hidden_size: int,
    state_matrix_dimension: int,
    training_steps: int,
    training_batch_size: int,
    evaluation_batch_count: int,
    learning_rate: float,
    random_seed: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    torch.manual_seed(random_seed)
    train_generator = torch.Generator(device=device).manual_seed(random_seed + 100_000)
    model = MultiQueryModel(method, config.vocabulary_size, model_hidden_size, state_matrix_dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    start = time.perf_counter()
    for _ in range(training_steps):
        model.train()
        batch = generate_multiquery_batch(config, training_batch_size, device, train_generator)
        logits, _ = model(batch.input_ids)
        loss = sequence_loss(logits, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    evaluation_loss, evaluation_accuracy = evaluate(
        model,
        config,
        training_batch_size,
        evaluation_batch_count,
        device,
        random_seed + 200_000,
    )
    return {
        "method": method,
        "random_seed": random_seed,
        "number_of_key_value_pairs": config.number_of_key_value_pairs,
        "sequence_length": config.sequence_length,
        "vocabulary_size": config.vocabulary_size,
        "model_hidden_size": model_hidden_size,
        "state_matrix_dimension": state_matrix_dimension,
        "training_steps": training_steps,
        "training_batch_size": training_batch_size,
        "evaluation_batch_count": evaluation_batch_count,
        "learning_rate": learning_rate,
        "final_evaluation_loss": evaluation_loss,
        "final_evaluation_accuracy": evaluation_accuracy,
        "steps_per_second": training_steps / max(time.perf_counter() - start, 1e-9),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Delta and OLA on a minimal multi-query task.")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--methods", default="delta,ola")
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--training-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-count", type=int, default=10)
    parser.add_argument("--vocabulary-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--number-of-key-value-pairs", type=int, default=2)
    parser.add_argument("--power-a", type=float, default=0.01)
    parser.add_argument("--random-non-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-hidden-size", type=int, default=32)
    parser.add_argument("--state-matrix-dimension", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-jsonl", type=Path, default=Path("runs/ola_mqar/multiquery_results.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/ola_mqar/multiquery_results.csv"))
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(methods) - {"delta", "ola"})
    if invalid:
        raise ValueError(f"unsupported methods: {invalid}")
    return methods


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = MultiQueryConfig(
        vocabulary_size=args.vocabulary_size,
        sequence_length=args.sequence_length,
        number_of_key_value_pairs=args.number_of_key_value_pairs,
        power_a=args.power_a,
        random_non_queries=args.random_non_queries,
    )
    rows = []
    for method in _parse_csv_strings(args.methods):
        for random_seed in _parse_csv_ints(args.seeds):
            row = train_once(
                method=method,
                config=config,
                model_hidden_size=args.model_hidden_size,
                state_matrix_dimension=args.state_matrix_dimension,
                training_steps=args.training_steps,
                training_batch_size=args.training_batch_size,
                evaluation_batch_count=args.evaluation_batch_count,
                learning_rate=args.learning_rate,
                random_seed=random_seed,
                device=device,
            )
            rows.append(row)
            append_jsonl(args.out_jsonl, row)
            write_csv(args.out_csv, rows)
            print(
                f"method={row['method']} random_seed={row['random_seed']} "
                f"final_evaluation_loss={row['final_evaluation_loss']:.4f} "
                f"final_evaluation_accuracy={row['final_evaluation_accuracy']:.4f} "
                f"steps_per_second={row['steps_per_second']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
