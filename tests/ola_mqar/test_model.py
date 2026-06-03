import torch

from experiments.ola_mqar.model import MQARModel


def test_delta_and_ola_models_return_vocab_logits():
    input_ids = torch.randint(0, 64, (3, 12))

    for method in ["delta", "ola"]:
        model = MQARModel(
            method=method,
            vocab_size=64,
            d_model=16,
            state_dim=8,
        )
        logits, aux = model(input_ids)

        assert logits.shape == (3, 12, 64)
        assert isinstance(aux, dict)
