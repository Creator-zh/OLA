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
