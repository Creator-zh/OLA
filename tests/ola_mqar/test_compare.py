from experiments.ola_mqar.compare import format_comparison


def test_format_comparison_reports_ola_minus_delta_gaps():
    text = format_comparison(
        delta={"eval_loss": 1.0, "eval_acc": 0.5, "steps_per_sec": 10.0},
        ola={"eval_loss": 1.2, "eval_acc": 0.4, "steps_per_sec": 8.0},
    )

    assert "ola_minus_delta_loss=0.2000" in text
    assert "ola_minus_delta_acc=-0.1000" in text
