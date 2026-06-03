from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


def peak_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def throughput_metrics(tokens: int, steps: int, elapsed_seconds: float) -> dict[str, float]:
    elapsed_seconds = max(float(elapsed_seconds), 1e-9)
    return {
        "tokens_per_second": float(tokens) / elapsed_seconds,
        "steps_per_second": float(steps) / elapsed_seconds,
        "seconds_per_step": elapsed_seconds / max(float(steps), 1.0),
    }


@dataclass(frozen=True)
class TimedBlock:
    elapsed_seconds: float


@contextmanager
def timed_block(device: torch.device) -> Iterator[TimedBlock]:
    synchronize(device)
    start = time.perf_counter()
    block = TimedBlock(elapsed_seconds=0.0)
    try:
        yield block
    finally:
        synchronize(device)
        object.__setattr__(block, "elapsed_seconds", time.perf_counter() - start)

