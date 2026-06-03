from experiments.ola_mqar.run_sweep import SweepConfig, build_jobs


def test_build_jobs_covers_methods_num_pairs_and_seeds():
    config = SweepConfig(
        methods=("delta", "ola"),
        num_pairs=(2, 4),
        seeds=(1, 2),
        steps=10,
        eval_interval=5,
        batch_size=8,
        eval_batches=2,
        vocab_size=64,
        d_models=(16, 32),
        state_dim=8,
        lr=0.003,
        device="cpu",
    )

    jobs = build_jobs(config)

    assert len(jobs) == 16
    assert jobs[0]["method"] == "delta"
    assert jobs[0]["num_pairs"] == 2
    assert jobs[0]["d_model"] == 16
    assert jobs[0]["seed"] == 1
    assert jobs[-1]["method"] == "ola"
    assert jobs[-1]["num_pairs"] == 4
    assert jobs[-1]["d_model"] == 32
    assert jobs[-1]["seed"] == 2
