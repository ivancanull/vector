from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn


DEFAULT_HIDDEN_DIMS = (64, 64, 64, 32, 16)
DEFAULT_KERNEL_SIZE = (32, 32, 32)
DEFAULT_STRIDE = (3, 3, 3)
DEFAULT_PADDING = (0, 1, 0)
DEFAULT_DILATION = (1, 1, 1)
DEFAULT_OUTPUT_PADDING = (0, 1, 0)
DEFAULT_DROPOUT = 0.3


@dataclass(frozen=True)
class ResponseTCNConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    output_channels: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"r": (2, 401), "i": (2, 401)}
    )
    kernel_size: tuple[int, ...] = DEFAULT_KERNEL_SIZE
    stride: tuple[int, ...] = DEFAULT_STRIDE
    padding: tuple[int, ...] = DEFAULT_PADDING
    dilation: tuple[int, ...] = DEFAULT_DILATION
    output_padding: tuple[int, ...] = DEFAULT_OUTPUT_PADDING
    dropout: float = DEFAULT_DROPOUT

    @classmethod
    def from_mapping(cls, values: Mapping[str, object], *, input_dim: int | None = None) -> "ResponseTCNConfig":
        raw_input_dim = input_dim if input_dim is not None else values.get("input_dim")
        if raw_input_dim is None:
            raise ValueError("ResponseTCNConfig requires input_dim")

        tcns = values.get("tcns", {})
        if not isinstance(tcns, Mapping):
            raise ValueError("model tcns config must be a mapping")

        output_channels = values.get("output_channels", {"r": (2, 401), "i": (2, 401)})
        if not isinstance(output_channels, Mapping):
            raise ValueError("output_channels must be a mapping")

        return cls(
            input_dim=int(raw_input_dim),
            hidden_dims=tuple(int(v) for v in values.get("hidden_dims", DEFAULT_HIDDEN_DIMS)),
            output_channels={
                str(name): _as_pair(spec, name=str(name))
                for name, spec in output_channels.items()
            },
            kernel_size=_as_int_tuple(tcns.get("kernel_size", DEFAULT_KERNEL_SIZE), "kernel_size"),
            stride=_as_int_tuple(tcns.get("stride", DEFAULT_STRIDE), "stride"),
            padding=_as_int_tuple(tcns.get("padding", DEFAULT_PADDING), "padding"),
            dilation=_as_int_tuple(tcns.get("dilation", DEFAULT_DILATION), "dilation"),
            output_padding=_as_int_tuple(tcns.get("output_padding", DEFAULT_OUTPUT_PADDING), "output_padding"),
            dropout=float(values.get("dropout", DEFAULT_DROPOUT)),
        )

    @property
    def output_length(self) -> int:
        lengths = {shape[1] for shape in self.output_channels.values()}
        if len(lengths) != 1:
            raise ValueError("all output channels must use the same frequency length")
        return int(next(iter(lengths)))

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if len(self.hidden_dims) != 5:
            raise ValueError("hidden_dims must contain exactly 5 values")
        n_conv = len(self.hidden_dims[-4:]) - 1
        for name, values in {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "output_padding": self.output_padding,
        }.items():
            if len(values) != n_conv:
                raise ValueError(f"{name} must contain exactly {n_conv} values")
        if set(self.output_channels) != {"r", "i"}:
            raise ValueError("output_channels must contain exactly 'r' and 'i'")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ResponseTCN(nn.Module):
    """Geometry-to-response network with an MLP encoder and transposed-conv decoder."""

    def __init__(self, config: ResponseTCNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        backbone_dims = (config.input_dim, *config.hidden_dims[:3])
        self.backbone = nn.ModuleList()
        for in_dim, out_dim in zip(backbone_dims[:-1], backbone_dims[1:]):
            self.backbone.append(nn.Linear(in_dim, out_dim))
            self.backbone.append(nn.ReLU())
            if config.dropout > 0:
                self.backbone.append(nn.Dropout(config.dropout))

        upsampling_dims = config.hidden_dims[-4:]
        self.upsampling = nn.ModuleList([nn.Unflatten(-1, (upsampling_dims[0], 1))])
        for i, (in_channels, out_channels) in enumerate(zip(upsampling_dims[:-1], upsampling_dims[1:])):
            self.upsampling.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size[i],
                    stride=config.stride[i],
                    padding=config.padding[i],
                    dilation=config.dilation[i],
                    output_padding=config.output_padding[i],
                )
            )
            self.upsampling.append(nn.ReLU())
            if config.dropout > 0:
                self.upsampling.append(nn.Dropout(config.dropout))

        self.output_heads = nn.ModuleDict(
            {
                name: nn.Conv1d(upsampling_dims[-1], out_channels, kernel_size=1)
                for name, (out_channels, _) in config.output_channels.items()
            }
        )
        self.names = list(self.output_heads.keys())
        self._init_weights()
        self._validate_output_length()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for layer in self.backbone:
            x = layer(x)
        for layer in self.upsampling:
            x = layer(x)

        outputs: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            out = head(x)
            expected = self.config.output_channels[name]
            if out.shape[-2:] != expected:
                raise RuntimeError(f"head {name} produced shape {tuple(out.shape[-2:])}, expected {expected}")
            outputs[name] = out
        return outputs

    def freeze(self) -> None:
        return None

    def unfreeze(self) -> None:
        return None

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.ConvTranspose1d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_output_length(self) -> None:
        length = 1
        for kernel, stride, padding, dilation, output_padding in zip(
            self.config.kernel_size,
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.config.output_padding,
        ):
            length = calculate_conv_transpose1d_length(
                length,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=output_padding,
            )
        if length != self.config.output_length:
            raise ValueError(f"transposed-conv stack outputs length {length}, expected {self.config.output_length}")


def calculate_conv_transpose1d_length(
    input_length: int,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    output_padding: int = 0,
) -> int:
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


def _as_int_tuple(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(int(v) for v in values)


def _as_pair(values: object, *, name: str) -> tuple[int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"output channel {name} must be a 2-item list or tuple")
    return int(values[0]), int(values[1])
