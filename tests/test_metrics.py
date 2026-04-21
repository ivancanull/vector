from pathlib import Path

from worker.metrics import compute_metrics


def test_compute_basic_s11_s21_metrics() -> None:
    touchstone = Path(__file__).parent / "data" / "sample.s2p"
    output = compute_metrics(touchstone, ["max_s11_db", "min_s21_db"])
    assert abs(output["max_s11_db"] - (-7.9588)) < 0.05
    assert abs(output["min_s21_db"] - (-10.4576)) < 0.05
