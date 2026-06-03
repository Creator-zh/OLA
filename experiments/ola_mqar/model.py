from __future__ import annotations

import torch
import torch.nn as nn

from experiments.ola_mqar.layers import DeltaMixer, OLAMixer


class MQARModel(nn.Module):
    def __init__(
        self,
        method: str,
        vocab_size: int,
        d_model: int = 128,
        state_dim: int = 32,
    ):
        super().__init__()
        if method not in {"delta", "ola"}:
            raise ValueError(f"method must be 'delta' or 'ola', got {method!r}")

        self.method = method
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mixer = DeltaMixer(d_model, state_dim) if method == "delta" else OLAMixer(d_model, state_dim)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embedding(input_ids)
        mixed = self.mixer(x)
        final_hidden = self.norm(mixed.hidden_states[:, -1])
        logits = self.lm_head(final_hidden)
        return logits, mixed.aux
