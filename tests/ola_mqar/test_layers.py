import torch

from experiments.ola_mqar.layers import OLAMixer


def test_ola_cayley_rotation_is_orthogonal_for_sampled_vectors():
    mixer = OLAMixer(d_model=16, state_dim=8)
    k_erase = torch.randn(2, 8)
    k_write = torch.randn(2, 8)
    eta = torch.full((2,), 0.2)

    rotation = mixer.cayley_rotation(k_erase, k_write, eta)
    identity = torch.eye(8).expand(2, 8, 8)

    error = torch.linalg.matrix_norm(rotation.transpose(-1, -2) @ rotation - identity)
    assert torch.max(error).item() < 1e-4


def test_ola_low_rank_state_rotation_matches_explicit_cayley_rotation():
    mixer = OLAMixer(d_model=16, state_dim=8)
    state = torch.randn(2, 8, 8)
    k_erase = torch.randn(2, 8)
    k_write = torch.randn(2, 8)
    eta = torch.full((2,), 0.2)
    alpha = torch.full((2,), 0.9)

    rotation = mixer.cayley_rotation(k_erase, k_write, eta)
    explicit = alpha.view(-1, 1, 1) * torch.bmm(rotation, state)
    low_rank = mixer.apply_cayley_state_rotation(state, k_erase, k_write, eta, alpha)

    assert torch.testing.assert_close(low_rank, explicit, rtol=1e-4, atol=1e-5) is None
