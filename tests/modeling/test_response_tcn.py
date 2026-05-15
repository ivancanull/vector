from __future__ import annotations

import pytest
import torch

from modeling import ResponseTCN, ResponseTCNConfig, calculate_conv_transpose1d_length


def test_response_tcn_forward_shapes() -> None:
    model = ResponseTCN(ResponseTCNConfig(input_dim=8, dropout=0.0))
    outputs = model(torch.zeros(3, 8))

    assert set(outputs) == {"r", "i"}
    assert outputs["r"].shape == (3, 2, 401)
    assert outputs["i"].shape == (3, 2, 401)


def test_response_tcn_output_length_matches_config() -> None:
    length = 1
    for kernel, stride, padding, dilation, output_padding in zip(
        (32, 32, 32),
        (3, 3, 3),
        (0, 1, 0),
        (1, 1, 1),
        (0, 1, 0),
    ):
        length = calculate_conv_transpose1d_length(
            length,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
        )

    assert length == 401


def test_response_tcn_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="hidden_dims"):
        ResponseTCN(ResponseTCNConfig(input_dim=8, hidden_dims=(64, 64), dropout=0.0))

    with pytest.raises(ValueError, match="outputs length"):
        ResponseTCN(
            ResponseTCNConfig(
                input_dim=8,
                output_channels={"r": (2, 400), "i": (2, 400)},
                dropout=0.0,
            )
        )
