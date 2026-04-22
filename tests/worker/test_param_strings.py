from worker.param_strings import (
    build_spiral_inductor_export_path,
    build_spiral_inductor_param_string,
    str_p,
)


def test_build_spiral_inductor_param_string_matches_recorded_script() -> None:
    assert (
        build_spiral_inductor_param_string(55, 2, 1.5, 15)
        == "Din=\\'55um\\' H=\\'2um\\' S=\\'1.5um\\' W=\\'15um\\'"
    )


def test_build_spiral_inductor_export_path_matches_recorded_script() -> None:
    assert build_spiral_inductor_export_path(55, 1.5, 15).endswith(
        "spiral_inductor_example_HFSSDesign0_Din_55_S_1d5_W_15.s2p"
    )


def test_str_p_preserves_integers_and_rewrites_decimals() -> None:
    assert str_p(55) == "55"
    assert str_p(1.5) == "1d5"
