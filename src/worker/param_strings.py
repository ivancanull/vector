from __future__ import annotations


def str_p(value: int | float, *, precision: int | None = None) -> str:
    if precision is None:
        text = str(value)
    else:
        text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
    return text.replace(".", "d")


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
    counter_width: int = 0,
    value_precision: int | None = None,
    output_dir: str = "D:/ansys/spiral_inductor_s_parameters_export",
    project_name: str = "spiral_inductor_example",
    design_prefix: str = "HFSSDesign",
    extension: str = ".s2p",
) -> str:
    return output_dir + "/" + build_spiral_inductor_export_filename(
        Din,
        S,
        W,
        counter=counter,
        counter_width=counter_width,
        value_precision=value_precision,
        project_name=project_name,
        design_prefix=design_prefix,
        extension=extension,
    )


def build_spiral_inductor_export_filename(
    Din: int | float,
    S: int | float,
    W: int | float,
    *,
    counter: int = 0,
    counter_width: int = 0,
    value_precision: int | None = None,
    project_name: str = "spiral_inductor_example",
    design_prefix: str = "HFSSDesign",
    extension: str = ".s2p",
) -> str:
    counter_text = str(counter).zfill(counter_width) if counter_width > 0 else str(counter)
    return (
        project_name
        + "_"
        + design_prefix
        + counter_text
        + "_Din_"
        + str_p(Din, precision=value_precision)
        + "_S_"
        + str_p(S, precision=value_precision)
        + "_W_"
        + str_p(W, precision=value_precision)
        + extension
    )
