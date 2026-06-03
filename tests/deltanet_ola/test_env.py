import torch


def test_collect_run_metadata_reads_fla_commit_when_submodule_exists():
    from experiments.deltanet_ola.env import collect_run_metadata
    from experiments.deltanet_ola.fla_paths import fla_root

    metadata = collect_run_metadata(torch.device("cpu"), dtype="float32")

    if fla_root().exists():
        assert metadata["fla_commit"] not in {"missing", "unknown", ""}
