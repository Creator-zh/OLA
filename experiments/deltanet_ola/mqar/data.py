from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MQARConfig:
    vocabulary_size: int = 8192
    sequence_length: int = 512
    number_of_key_value_pairs: int = 64
    power_a: float = 0.01
    random_non_queries: bool = True


@dataclass(frozen=True)
class MQARBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_mask: torch.Tensor


def validate_config(config: MQARConfig) -> None:
    if config.sequence_length % 2 != 0:
        raise ValueError("sequence_length must be even.")
    if config.number_of_key_value_pairs * 4 > config.sequence_length:
        raise ValueError("sequence_length must fit key-value context and query slots.")
    if config.vocabulary_size <= config.sequence_length:
        raise ValueError("vocabulary_size must be larger than sequence_length.")
    if config.number_of_key_value_pairs >= config.vocabulary_size // 2:
        raise ValueError("number_of_key_value_pairs must fit key and value token ranges.")
    if config.power_a <= 0:
        raise ValueError("power_a must be positive.")


def generate_mqar_batch(
    config: MQARConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> MQARBatch:
    validate_config(config)
    examples = torch.zeros(batch_size, config.sequence_length + 1, dtype=torch.long, device=device)
    raw_labels = torch.full((batch_size, config.sequence_length + 1), -100, dtype=torch.long, device=device)
    key_pool = torch.arange(1, config.vocabulary_size // 2, device=device)
    value_pool = torch.arange(config.vocabulary_size // 2, config.vocabulary_size, device=device)
    context_length = config.number_of_key_value_pairs * 2
    gap_space = (config.sequence_length - context_length) // 2
    gap_weights = config.power_a * torch.arange(1, gap_space + 1, device=device, dtype=torch.float32).pow(
        config.power_a - 1
    )
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
    return MQARBatch(input_ids=input_ids, labels=labels, query_mask=labels != -100)


def sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def query_accuracy(logits: torch.Tensor, labels: torch.Tensor, query_mask: torch.Tensor) -> float:
    correct = ((logits.argmax(dim=-1) == labels) & query_mask).sum().item()
    total = query_mask.sum().item()
    return correct / max(total, 1)

