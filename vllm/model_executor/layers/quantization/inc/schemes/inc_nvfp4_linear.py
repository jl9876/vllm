# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.fusion.quant_activation import expose_input_quant_key
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs

from .inc_scheme import INCLinearScheme

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig


class INCNvfp4LinearMethod(INCLinearScheme):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.group_size = layer_config.group_size
        self.kernel = init_nvfp4_linear_kernel()

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        layer.register_parameter(
            "weight_packed",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // self.group_size,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        set_weight_attrs(weight_global_scale, {"needs_scalar_to_array": True})
        layer.register_parameter("weight_global_scale", weight_global_scale)
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        set_weight_attrs(input_global_scale, {"needs_scalar_to_array": True})
        layer.register_parameter("input_global_scale", input_global_scale)
        expose_input_quant_key(layer, self.kernel)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = layer.weight_packed
        del layer.weight_packed

        weight_global_scale_inv = layer.weight_global_scale.max().to(torch.float32)
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale_inv, requires_grad=False
        )
        input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(
            1.0 / input_global_scale_inv, requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(
            input_global_scale_inv, requires_grad=False
        )
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale,
            requires_grad=False,
        )
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)