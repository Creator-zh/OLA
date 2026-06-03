from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MQARConfig:
    vocab_size: int = 128
    input_seq_len: int = 64
    num_kv_pairs: int = 4
    power_a: float = 0.01
    random_non_queries: bool = True


@dataclass(frozen=True)
class MQARBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_mask: torch.Tensor


def _validate_config(config: MQARConfig) -> None:
    if config.input_seq_len % 2 != 0:
        raise ValueError("input_seq_len must be even.")
    if config.vocab_size <= config.input_seq_len:
        raise ValueError("vocab_size must be larger than input_seq_len.")
    if config.num_kv_pairs >= config.vocab_size // 2:
        raise ValueError("num_kv_pairs must fit inside the key and value token ranges.")
    if config.num_kv_pairs * 4 > config.input_seq_len:
        raise ValueError("MQAR needs room for key-value context plus query positions.")
    if config.power_a <= 0:
        raise ValueError("power_a must be positive.")


def generate_mqar_batch(
    config: MQARConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> MQARBatch:
    """Generate Zoology-style multi-query associative recall sequences."""
    _validate_config(config)

    input_ids = torch.zeros(batch_size, config.input_seq_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, config.input_seq_len), -100, dtype=torch.long, device=device)
    context_size = config.num_kv_pairs * 2
    key_pool = torch.arange(1, config.vocab_size // 2, device=device)
    value_pool = torch.arange(config.vocab_size // 2, config.vocab_size, device=device)
    gap_space = (config.input_seq_len - context_size) // 2
    gap_weights = config.power_a * torch.arange(
        1,
        gap_space + 1,
        device=device,
        dtype=torch.float32,
    ).pow(config.power_a - 1)
    gap_weights = gap_weights / gap_weights.sum()

    for b in range(batch_size):
        keys = key_pool[torch.randperm(key_pool.numel(), device=device, generator=generator)[: config.num_kv_pairs]]
        values = value_pool[torch.randperm(value_pool.numel(), device=device, generator=generator)[: config.num_kv_pairs]]
        gaps = torch.multinomial(gap_weights, config.num_kv_pairs, replacement=False, generator=generator)
        query_positions = context_size + gaps * 2

        input_ids[b, 0:context_size:2] = keys
        input_ids[b, 1:context_size:2] = values
        input_ids[b, query_positions] = keys
        labels[b, query_positions] = values

    if config.random_non_queries:
        zero_mask = input_ids == 0
        random_tokens = torch.randint(
            config.vocab_size,
            size=input_ids.shape,
            device=device,
            generator=generator,
        )
        input_ids[zero_mask] = random_tokens[zero_mask]

    return MQARBatch(
        input_ids=input_ids,
        labels=labels,
        query_mask=labels != -100,
    )
