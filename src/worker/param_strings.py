from __future__ import annotations


def str_p(value: int | float) -> str:
    return str(value).replace(".", "d")


def build_spiral_inductor_param_string(
    Din: int | float,
    H: int | float,
    S: int | float,
    W: int | float,
    unit: str = "um",
) -> str:
    return (
        "Din=\\'"
        + str(Din)
        + unit
        + "\\' H=\\'"
        + str(H)
        + unit
        + "\\' S=\\'"
        + str(S)
        + unit
        + "\\' W=\\'"
        + str(W)
        + unit
        + "\\'"
    )


def build_spiral_inductor_export_path(
    Din: int | float,
    S: int | float,
    W: int | float,
    *,
    counter: int = 0,
    output_dir: str = "D:/ansys/spiral_inductor_s_parameters_export",
    project_name: str = "spiral_inductor_example",
    design_prefix: str = "HFSSDesign",
    extension: str = ".s2p",
) -> str:
    return (
        output_dir
        + "/"
        + project_name
        + "_"
        + design_prefix
        + str(counter)
        + "_Din_"
        + str_p(Din)
        + "_S_"
        + str_p(S)
        + "_W_"
        + str_p(W)
        + extension
    )


if __name__ == "__main__":
    print(build_spiral_inductor_param_string(55, 2, 1.5, 15))
    print(build_spiral_inductor_export_path(55, 1.5, 15))
