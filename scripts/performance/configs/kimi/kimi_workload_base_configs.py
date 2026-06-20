# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workload base presets for Kimi-K2 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_KIMI_K2_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


KIMI_K2_PRETRAIN_CONFIG_GB300 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    micro_batch_size=2,
    cuda_graph_scope=[],
    recompute_modules=["mla_up_proj"],
)
KIMI_K2_PRETRAIN_CONFIG_GB300_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4 = KIMI_K2_PRETRAIN_CONFIG_GB300


KIMI_K2_PRETRAIN_CONFIG_GB200 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
KIMI_K2_PRETRAIN_CONFIG_GB200_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB200
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_GB200
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB200

KIMI_K2_PRETRAIN_CONFIG_VR200_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB200_BF16
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX


# 128-GPU VR200 variant (config_variant="128gpu"): half the GPUs of the 256-GPU
# baseline. PP=2 / MBS=1 / no CUDA graph, with selective recompute to absorb the
# heavier per-stage memory (PP=2 doubles layers-per-stage vs the PP=4 baseline).
# NOTE: the Kimi-K2 pipeline-layout map only defines PP=2 paired with VP=8, so
# virtual_pipeline_model_parallel_size MUST be 8 here. EP=64 keeps DP=1
# (128 / (PP=2 * EP=64)).
KIMI_K2_PRETRAIN_CONFIG_VR200_128GPU = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=128,
    global_batch_size=1024,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    expert_model_parallel_size=64,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj", "moe_act"],
    cuda_graph_impl="none",
)
KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_128GPU = KIMI_K2_PRETRAIN_CONFIG_VR200_128GPU
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_128GPU = KIMI_K2_PRETRAIN_CONFIG_VR200_128GPU


KIMI_K2_PRETRAIN_CONFIG_B300 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=16,
    global_batch_size=4096,
    micro_batch_size=2,
    recompute_modules=["mla_up_proj"],
    moe_a2a_overlap=False,
)
KIMI_K2_PRETRAIN_CONFIG_B300_BF16 = KIMI_K2_PRETRAIN_CONFIG_B300
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_B300
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_B300


KIMI_K2_PRETRAIN_CONFIG_B200 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=16,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_a2a_overlap=False,
)
KIMI_K2_PRETRAIN_CONFIG_B200_BF16 = KIMI_K2_PRETRAIN_CONFIG_B200
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_B200
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_B200


KIMI_K2_PRETRAIN_CONFIG_H100 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
    moe_a2a_overlap=False,
    pp_layout="Et|(tt|)*30L",
)
KIMI_K2_PRETRAIN_CONFIG_H100_BF16 = KIMI_K2_PRETRAIN_CONFIG_H100
KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_H100
KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC = KIMI_K2_PRETRAIN_CONFIG_H100



# GB300-based VR200 configs (config_variant="gb300"): VR200 adopts the GB300
# recipe wholesale, plus the VR200 CuteDSL fused grouped MLP override. The source
# GB300 configs are unchanged (replace() returns a copy).
KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_GB300 = replace(
    KIMI_K2_PRETRAIN_CONFIG_GB300_BF16, cutedsl_fused_grouped_mlp=True
)
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_CS_GB300 = replace(
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS, cutedsl_fused_grouped_mlp=True
)
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_GB300 = replace(
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX, cutedsl_fused_grouped_mlp=True
)
KIMI_K2_PRETRAIN_CONFIG_VR200_NVFP4_GB300 = replace(
    KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4, cutedsl_fused_grouped_mlp=True
)
__all__ = [
    "KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_GB300",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_CS_GB300",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_GB300",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_NVFP4_GB300",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_128GPU",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_128GPU",
    "KIMI_K2_PRETRAIN_CONFIG_B300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_B200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_H100_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC",
]
