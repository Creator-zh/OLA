from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MQARSweepConfig:
    methods: tuple[str, ...] = ("fla_delta", "ola")
    vocabulary_size: int = 8192
    sequence_length: int = 512
    number_of_key_value_pairs: int = 64
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


def build_mqar_jobs(config: MQARSweepConfig) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for method in config.methods:
        for model_hidden_size in config.model_hidden_sizes:
            state_matrix_dimension = config.state_matrix_dimension or model_hidden_size
            for random_seed in config.random_seeds:
                jobs.append(
                    {
                        "method": method,
                        "vocabulary_size": config.vocabulary_size,
                        "sequence_length": config.sequence_length,
                        "number_of_key_value_pairs": config.number_of_key_value_pairs,
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

