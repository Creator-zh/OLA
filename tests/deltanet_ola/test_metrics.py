import torch


def test_peak_memory_bytes_is_cpu_safe():
    from experiments.deltanet_ola.metrics import peak_memory_bytes

    assert peak_memory_bytes(torch.device("cpu")) == 0


def test_throughput_metrics_include_tokens_and_steps():
    from experiments.deltanet_ola.metrics import throughput_metrics

    metrics = throughput_metrics(tokens=2048, steps=4, elapsed_seconds=2.0)

    assert metrics["tokens_per_second"] == 1024.0
    assert metrics["steps_per_second"] == 2.0
    assert metrics["seconds_per_step"] == 0.5
