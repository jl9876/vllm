# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class INCNvfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe) -> None:
        super().__init__(moe)
        self.group_size = 16
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        def register(name: str, data: torch.Tensor, scale_type=None) -> None:
            parameter = torch.nn.Parameter(data, requires_grad=False)
            layer.register_parameter(name, parameter)
            attrs = dict(extra_weight_attrs)
            if scale_type is not None:
                attrs["quant_method"] = scale_type.value
            set_weight_attrs(parameter, attrs)

        register(
            "w13_weight_packed",
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
        )
        register(
            "w2_weight_packed",
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
        )
        register(
            "w13_weight_scale",
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            FusedMoeWeightScaleSupported.GROUP,
        )
        register(
            "w2_weight_scale",
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            FusedMoeWeightScaleSupported.GROUP,
        )
        register(
            "w13_weight_global_scale",
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            FusedMoeWeightScaleSupported.TENSOR,
        )
        register(
            "w2_weight_global_scale",
            torch.empty(num_experts, dtype=torch.float32),
            FusedMoeWeightScaleSupported.TENSOR,
        )
        register(
            "w13_input_global_scale",
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            FusedMoeWeightScaleSupported.TENSOR,
        )
        register(
            "w2_input_global_scale",
            torch.empty(num_experts, dtype=torch.float32),
            FusedMoeWeightScaleSupported.TENSOR,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0],
            layer.w13_weight_global_scale[:, 1],
        ):
            logger.warning_once(
                "NVFP4 gate and up projection global scales differ; "
                "accuracy may be affected."
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        converted = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=1.0 / w13_weight_global_scale,
            a13_scale=1.0 / layer.w13_input_global_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=1.0 / layer.w2_weight_global_scale,
            a2_scale=1.0 / layer.w2_input_global_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
        )
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = converted
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w2_weight_scale_2 = w2_scale_2
        layer.w13_input_scale = a13_scale
        layer.w2_input_scale = a2_scale

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses modular kernel initialization"
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )