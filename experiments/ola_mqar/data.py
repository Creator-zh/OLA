from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MQARConfig:
    vocab_size: int = 128
    num_pairs: int = 8
    query_token_id: int = 0

    @property
    def seq_len(self) -> int:
        return self.num_pairs * 2 + 2

    @property
    def key_start(self) -> int:
        return 1

    @property
    def value_start(self) -> int:
        return max(self.vocab_size // 2, self.key_start + self.num_pairs + 1)


@dataclass(frozen=True)
class MQARBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    value_positions: torch.Tensor


def _validate_config(config: MQARConfig) -> None:
    key_capacity = config.value_start - config.key_start
    value_capacity = config.vocab_size - config.value_start
    if config.query_token_id != 0:
        raise ValueError("The minimal experiment reserves token 0 for the query marker.")
    if config.num_pairs > key_capacity:
        raise ValueError(
            f"num_pairs={config.num_pairs} exceeds key capacity={key_capacity}; "
            "increase vocab_size or reduce num_pairs."
        )
    if config.num_pairs > value_capacity:
        raise ValueError(
            f"num_pairs={config.num_pairs} exceeds value capacity={value_capacity}; "
            "increase vocab_size or reduce num_pairs."
        )


def generate_mqar_batch(
    config: MQARConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> MQARBatch:
    """Generate single-query associative recall samples: k1 v1 ... <query> kq -> vq."""
    _validate_config(config)

    input_ids = torch.empty(batch_size, config.seq_len, dtype=torch.long, device=device)
    labels = torch.empty(batch_size, dtype=torch.long, device=device)
    query_positions = torch.full((batch_size,), config.seq_len - 1, dtype=torch.long, device=device)
    value_positions = torch.empty(batch_size, dtype=torch.long, device=device)

    key_pool = torch.arange(config.key_start, config.value_start, device=device)
    value_pool = torch.arange(config.value_start, config.vocab_size, device=device)

    for b in range(batch_size):
        keys = key_pool[torch.randperm(key_pool.numel(), device=device, generator=generator)[: config.num_pairs]]
        values = value_pool[torch.randperm(value_pool.numel(), device=device, generator=generator)[: config.num_pairs]]
        target_idx = torch.randint(0, config.num_pairs, (1,), device=device, generator=generator).item()

        input_ids[b, 0 : config.num_pairs * 2 : 2] = keys
        input_ids[b, 1 : config.num_pairs * 2 : 2] = values
        input_ids[b, -2] = config.query_token_id
        input_ids[b, -1] = keys[target_idx]

        value_pos = target_idx * 2 + 1
        value_positions[b] = value_pos
        labels[b] = input_ids[b, value_pos]

    return MQARBatch(
        input_ids=input_ids,
        labels=labels,
        query_positions=query_positions,
        value_positions=value_positions,
    )
