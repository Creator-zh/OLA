from __future__ import annotations

import importlib

import torch
import torch.nn as nn

from experiments.deltanet_ola.fla_paths import ensure_fla_on_path
from experiments.ola_mqar.layers import OLAMixer


class OLAMixerAdapter(nn.Module):
    def __init__(self, hidden_size: int, state_matrix_dimension: int):
        super().__init__()
        self.mixer = OLAMixer(hidden_size, state_matrix_dimension)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = self.mixer(hidden_states)
        return mixed.hidden_states, mixed.aux


class FLADeltaNetAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_matrix_dimension: int,
        delta_num_heads: int,
        delta_use_short_conv: bool,
    ):
        super().__init__()
        ensure_fla_on_path()
        try:
            fla_layers = importlib.import_module("fla.layers")
        except Exception as exc:
            raise ImportError(
                "Could not import FLA DeltaNet. Initialize the submodule and install FLA dependencies from "
                "3rdparty/flash-linear-attention before running method='fla_delta'."
            ) from exc

        self.mixer = fla_layers.DeltaNet(
            mode="chunk",
            hidden_size=hidden_size,
            expand_k=state_matrix_dimension / hidden_size,
            expand_v=state_matrix_dimension / hidden_size,
            num_heads=delta_num_heads,
            use_beta=True,
            use_gate=False,
            use_short_conv=delta_use_short_conv,
            qk_activation="silu",
            qk_norm="l2",
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed_hidden, _, _ = self.mixer(hidden_states)
        return mixed_hidden, {}


def build_mixer(
    method: str,
    hidden_size: int,
    state_matrix_dimension: int,
    delta_num_heads: int,
    delta_use_short_conv: bool,
) -> nn.Module:
    if method == "ola":
        return OLAMixerAdapter(hidden_size, state_matrix_dimension)
    if method == "fla_delta":
        return FLADeltaNetAdapter(
            hidden_size=hidden_size,
            state_matrix_dimension=state_matrix_dimension,
            delta_num_heads=delta_num_heads,
            delta_use_short_conv=delta_use_short_conv,
        )
    raise ValueError(f"unsupported method: {method!r}")

