from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MixerOutput:
    hidden_states: torch.Tensor
    aux: dict[str, torch.Tensor]


def _outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left.unsqueeze(-1) * right.unsqueeze(-2)


class DeltaMixer(nn.Module):
    """Small pure-PyTorch DeltaNet recurrence for synthetic experiments."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.q_proj = nn.Linear(d_model, state_dim, bias=False)
        self.k_proj = nn.Linear(d_model, state_dim, bias=False)
        self.v_proj = nn.Linear(d_model, state_dim, bias=False)
        self.beta_proj = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(state_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> MixerOutput:
        batch_size, seq_len, _ = x.shape
        q = F.normalize(self.q_proj(x), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x)).squeeze(-1)

        state = x.new_zeros(batch_size, self.state_dim, self.state_dim)
        outputs = []
        for t in range(seq_len):
            kt = k[:, t]
            vt = v[:, t]
            bt = beta[:, t].unsqueeze(-1)
            erase = torch.bmm(kt.unsqueeze(1), state).squeeze(1)
            state = state - _outer(kt, bt * erase) + _outer(kt, bt * vt)
            ot = torch.bmm(q[:, t].unsqueeze(1), state).squeeze(1)
            outputs.append(ot)

        hidden = self.out_proj(torch.stack(outputs, dim=1))
        return MixerOutput(hidden_states=hidden, aux={"mean_beta": beta.mean().detach()})


class OLAMixer(nn.Module):
    """Orthogonal transition recurrence using a Cayley transform."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.q_proj = nn.Linear(d_model, state_dim, bias=False)
        self.k_erase_proj = nn.Linear(d_model, state_dim, bias=False)
        self.k_write_proj = nn.Linear(d_model, state_dim, bias=False)
        self.v_proj = nn.Linear(d_model, state_dim, bias=False)
        self.eta_proj = nn.Linear(d_model, 1)
        self.alpha_proj = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(state_dim, d_model, bias=False)

        nn.init.constant_(self.alpha_proj.bias, 2.0)

    def cayley_rotation(
        self,
        k_erase: torch.Tensor,
        k_write: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        k_erase = F.normalize(k_erase, dim=-1)
        k_write = F.normalize(k_write, dim=-1)
        a = eta.view(-1, 1, 1) * (_outer(k_erase, k_write) - _outer(k_write, k_erase))
        identity = torch.eye(self.state_dim, device=a.device, dtype=a.dtype).expand_as(a)
        return torch.linalg.solve(identity + a, identity - a)

    def apply_cayley_state_rotation(
        self,
        state: torch.Tensor,
        k_erase: torch.Tensor,
        k_write: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        k_erase = F.normalize(k_erase, dim=-1)
        k_write = F.normalize(k_write, dim=-1)
        scale = torch.sqrt(eta).unsqueeze(-1)
        u = torch.stack((scale * k_erase, -scale * k_write), dim=-1)
        v = torch.stack((scale * k_write, scale * k_erase), dim=-1)
        core = torch.eye(2, device=state.device, dtype=state.dtype).expand(state.shape[0], 2, 2)
        core = core + torch.bmm(v.transpose(1, 2), u)
        projected_state = torch.bmm(v.transpose(1, 2), state)
        solved = torch.linalg.solve(core, projected_state)
        rotated_state = state - 2.0 * torch.bmm(u, solved)
        return alpha.view(-1, 1, 1) * rotated_state

    def forward(self, x: torch.Tensor) -> MixerOutput:
        batch_size, seq_len, _ = x.shape
        q = F.normalize(self.q_proj(x), dim=-1)
        k_erase = self.k_erase_proj(x)
        k_write = F.normalize(self.k_write_proj(x), dim=-1)
        v = self.v_proj(x)
        eta = 0.5 * torch.sigmoid(self.eta_proj(x)).squeeze(-1)
        alpha = torch.sigmoid(self.alpha_proj(x)).squeeze(-1)

        state = x.new_zeros(batch_size, self.state_dim, self.state_dim)
        outputs = []
        first_rotation = None
        for t in range(seq_len):
            if first_rotation is None:
                rotation = self.cayley_rotation(k_erase[:, t], k_write[:, t], eta[:, t])
                first_rotation = rotation.detach()
            state = self.apply_cayley_state_rotation(state, k_erase[:, t], k_write[:, t], eta[:, t], alpha[:, t])
            state = state + _outer(k_write[:, t], v[:, t])
            ot = torch.bmm(q[:, t].unsqueeze(1), state).squeeze(1)
            outputs.append(ot)

        hidden = self.out_proj(torch.stack(outputs, dim=1))
        aux: dict[str, torch.Tensor] = {
            "mean_eta": eta.mean().detach(),
            "mean_alpha": alpha.mean().detach(),
        }
        if first_rotation is not None:
            identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype).expand_as(first_rotation)
            aux["orthogonality_error"] = torch.linalg.matrix_norm(
                first_rotation.transpose(-1, -2) @ first_rotation - identity,
            ).max().detach()
        return MixerOutput(hidden_states=hidden, aux=aux)
