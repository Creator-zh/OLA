import pytest
import torch


def test_ola_adapter_returns_sequence_hidden_states_on_cpu():
    from experiments.deltanet_ola.methods import build_mixer

    mixer = build_mixer(
        method="ola",
        hidden_size=16,
        state_matrix_dimension=8,
        delta_num_heads=2,
        delta_use_short_conv=False,
    )
    x = torch.randn(2, 5, 16)

    hidden, aux = mixer(x)

    assert hidden.shape == x.shape
    assert "mean_eta" in aux


def test_unknown_method_is_rejected():
    from experiments.deltanet_ola.methods import build_mixer

    with pytest.raises(ValueError, match="unsupported method"):
        build_mixer(
            method="unknown",
            hidden_size=16,
            state_matrix_dimension=8,
            delta_num_heads=2,
            delta_use_short_conv=False,
        )
