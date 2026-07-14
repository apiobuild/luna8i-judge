"""
Infra registry — self-hosted GPU provider pricing for scale projection.

Inclusion bar: ≥ 24 GB total VRAM per instance (enough to run a quantized 7B model).
Instances below this bar produce no useful throughput estimates and are omitted.

# Live-fetchable providers (public APIs, no auth required)
#
#   RunPod — GraphQL, no auth
#     Endpoint: https://api.runpod.io/graphql
#     Query: { gpuTypes { displayName memoryInGb securePrice communityPrice } }
#     Status: IMPLEMENTED
#
#   Vast.ai — REST, no auth for price queries (marketplace; prices vary per host)
#     Endpoint: https://cloud.vast.ai/api/v0/bundles/?verified=true
#     Docs: https://vast.ai/docs/api
#     Note: returns individual offers, not a canonical price list; needs aggregation (p25/median)
#
# Hardcoded providers (no viable public API without auth)
#
#   AWS — bulk JSON pricing files exist but require parsing ~50 MB manifests
#     Spot:      https://aws.amazon.com/ec2/spot/pricing/
#     On-demand: https://aws.amazon.com/ec2/pricing/on-demand/
#
#   GCP — pricing calculator API requires OAuth
#     On-demand + spot: https://cloud.google.com/compute/vm-instance-pricing
#
#   Lambda Labs — REST API exists but requires an API key
#     Docs: https://docs.lambdalabs.com/public-cloud/cloud-api/
#     Pricing page: https://lambdalabs.com/service/gpu-cloud
#
#   CoreWeave — pricing requires account; no public API
#     Pricing page: https://www.coreweave.com/gpu-cloud-pricing
#
# Update _PROVIDERS manually when hardcoded prices change.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from src.db import get_session
from src.schemas.db import InfraProviderRecord
from src.schemas.jobs import InfraProviderPricing

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0


# minimum per-GPU VRAM; filters out T4 (16 GB), V100 (16 GB), etc.
_MIN_TOTAL_VRAM_GB = 24


# Throughput lookup: (gpu_type, model_size_class) → decode tokens/sec PER SINGLE GPU (output tokens only)
#
# TPS: tokens per second produced by one physical GPU card under continuous batching.
# It is NOT a per-VM figure. VM throughput = value × provider.gpu_count.
#   e.g. aws_p4d_24xlarge (8× A100 40GB, 7B model): 2,000 × 8 = 16,000 tok/s
# instances_to_fit is a separate VRAM feasibility count and must NOT be used as a throughput multiplier.
#
# Derivation (memory-bandwidth-bound roofline model):
#   Reference: "Efficiently Scaling Transformer Inference", Pope et al. 2022
#              https://arxiv.org/abs/2211.05100
#   Formula:
#     single_request_ceiling = bandwidth_GB_s / model_size_GB
#     model_size_GB          = params_B × bytes_per_param   (fp16 → 2 bytes)
#     batch_throughput       = single_request_ceiling × batch_multiplier
#   The batch_multiplier accounts for vLLM continuous batching amortizing weight loads
#   across concurrent requests. We use 20× as a conservative estimate; real values
#   range 15–35× depending on concurrency, sequence length, and KV-cache pressure.
#   This multiplier is not sourced from a benchmark — it is an engineering estimate.
#   For real numbers run benchmark_throughput.py from the vLLM repo.
#
# GPU memory bandwidth (official NVIDIA data sheets):
#   A100 40GB SXM:  1,555 GB/s  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
#   A100 80GB SXM:  2,039 GB/s  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf
#   H100 80GB SXM:  3,350 GB/s  https://www.nvidia.com/en-us/data-center/h100/
#   L40S 48GB:        864 GB/s  https://www.nvidia.com/en-us/data-center/l40s/
#   L4 24GB:          300 GB/s  https://www.nvidia.com/en-us/data-center/l4/
#   RTX 4090 24GB:  1,008 GB/s  https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
#   B200 192GB:     8,000 GB/s  https://www.nvidia.com/en-us/data-center/b200/
#   B300 228GB:     8,000 GB/s  https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/
#
# Example (a100_40gb, 7b):
#   model_size = 7 × 2 = 14 GB
#   single_request_ceiling = 1555 / 14 ≈ 111 tok/s
#   batch_throughput = 111 × 20 ≈ 2,200 → rounded to 2,000 (conservative)
#
# Accuracy: ±50% real-world variance. Use only for order-of-magnitude scale projection.
# gpu_type matches the InfraProvider.gpu_type field below.
# model_size_class: "7b" | "13b" | "34b" | "70b"
_TOKENS_PER_SEC: dict[tuple[str, str], int] = {
    # a100_40gb: bandwidth=1555 GB/s  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
    # multiplier=20; 7b→14GB: 1555/14×20≈2200→2000
    ("a100_40gb", "7b"): 2_000,
    # 13b→26GB: 1555/26×20≈1200
    ("a100_40gb", "13b"): 1_200,
    # a100_80gb: bandwidth=2039 GB/s  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf
    # 7b→14GB: 2039/14×20≈2900→2800
    ("a100_80gb", "7b"): 2_800,
    # 13b→26GB: 2039/26×20≈1570→1500
    ("a100_80gb", "13b"): 1_500,
    # 34b→68GB: 2039/68×20≈600
    ("a100_80gb", "34b"): 600,
    # 70b→140GB: 2039/140×20≈290 (needs tensor parallelism across GPUs)
    ("a100_80gb", "70b"): 290,
    # h100_80gb: bandwidth=3350 GB/s  https://www.nvidia.com/en-us/data-center/h100/
    # 7b→14GB: 3350/14×20≈4800→4500
    ("h100_80gb", "7b"): 4_500,
    # 13b→26GB: 3350/26×20≈2580→2500
    ("h100_80gb", "13b"): 2_500,
    # 34b→68GB: 3350/68×20≈985→900
    ("h100_80gb", "34b"): 900,
    # 70b→140GB: 3350/140×20≈480
    ("h100_80gb", "70b"): 480,
    # h100_94gb (H100 NVL): bandwidth=3938 GB/s  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf
    # 7b→14GB: 3938/14×20≈5630→5500
    ("h100_94gb", "7b"): 5_500,
    # 13b→26GB: 3938/26×20≈3030→3000
    ("h100_94gb", "13b"): 3_000,
    # 34b→68GB: 3938/68×20≈1160→1100
    ("h100_94gb", "34b"): 1_100,
    # 70b→140GB: 3938/140×20≈562→550
    ("h100_94gb", "70b"): 550,
    # h200_141gb (H200 SXM): bandwidth=4800 GB/s  https://www.nvidia.com/en-us/data-center/h200/
    # 7b→14GB: 4800/14×20≈6860→6500
    ("h200_141gb", "7b"): 6_500,
    # 13b→26GB: 4800/26×20≈3690→3500
    ("h200_141gb", "13b"): 3_500,
    # 34b→68GB: 4800/68×20≈1412→1400
    ("h200_141gb", "34b"): 1_400,
    # 70b→140GB: 4800/140×20≈686→650
    ("h200_141gb", "70b"): 650,
    # h200_143gb (H200 NVL): bandwidth=4800 GB/s — same die as H200 SXM, lower TDP
    # https://www.nvidia.com/en-us/data-center/h200/
    # 7b→14GB: 4800/14×20≈6860→6500
    ("h200_143gb", "7b"): 6_500,
    # 13b→26GB: 4800/26×20≈3690→3500
    ("h200_143gb", "13b"): 3_500,
    # 34b→68GB: 4800/68×20≈1412→1400
    ("h200_143gb", "34b"): 1_400,
    # 70b→140GB: 4800/140×20≈686→650
    ("h200_143gb", "70b"): 650,
    # l40s_48gb: bandwidth=864 GB/s  https://www.nvidia.com/en-us/data-center/l40s/
    # 7b→14GB: 864/14×20≈1230→1200
    ("l40s_48gb", "7b"): 1_200,
    # 13b→26GB: 864/26×20≈665→600
    ("l40s_48gb", "13b"): 600,
    # 34b→68GB: 864/68×20≈254→250
    ("l40s_48gb", "34b"): 250,
    # l40_48gb: bandwidth=864 GB/s (L40 and L40S share nearly identical memory bandwidth)
    # https://www.nvidia.com/en-us/data-center/l40/
    # 7b→14GB: same as l40s
    ("l40_48gb", "7b"): 1_200,
    ("l40_48gb", "13b"): 600,
    ("l40_48gb", "34b"): 250,
    # l4_24gb: bandwidth=300 GB/s  https://www.nvidia.com/en-us/data-center/l4/
    # 7b→14GB: 300/14×20≈430→400
    ("l4_24gb", "7b"): 400,
    # 13b→26GB: 300/26×20≈230→200
    ("l4_24gb", "13b"): 200,
    # a40_48gb: bandwidth=696 GB/s  https://www.nvidia.com/en-us/data-center/a40/
    # 7b→14GB: 696/14×20≈994→950
    ("a40_48gb", "7b"): 950,
    # 13b→26GB: 696/26×20≈536→500
    ("a40_48gb", "13b"): 500,
    # 34b→68GB: 696/68×20≈205→200
    ("a40_48gb", "34b"): 200,
    # rtx_3090_24gb: bandwidth=936 GB/s  https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
    # 7b→14GB: 936/14×20≈1337→1300
    ("rtx_3090_24gb", "7b"): 1_300,
    # 13b→26GB: 936/26×20≈720→700
    ("rtx_3090_24gb", "13b"): 700,
    # rtx_3090_ti_24gb: bandwidth=1008 GB/s (same die as RTX 4090, binned)
    # https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
    # 7b→14GB: 1008/14×20≈1440→1400
    ("rtx_3090_ti_24gb", "7b"): 1_400,
    # 13b→26GB: 1008/26×20≈776→750
    ("rtx_3090_ti_24gb", "13b"): 750,
    # rtx_4090_24gb: bandwidth=1008 GB/s  https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
    # 7b→14GB: 1008/14×20≈1440→1400
    ("rtx_4090_24gb", "7b"): 1_400,
    # 13b→26GB: 1008/26×20≈776→750
    ("rtx_4090_24gb", "13b"): 750,
    # rtx_5090_32gb: bandwidth=1792 GB/s  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
    # 7b→14GB: 1792/14×20≈2560→2500
    ("rtx_5090_32gb", "7b"): 2_500,
    # 13b→26GB: 1792/26×20≈1378→1300
    ("rtx_5090_32gb", "13b"): 1_300,
    # b200_180gb: same die as B200 192GB (RunPod exposes 180GB); bandwidth=8000 GB/s  https://www.nvidia.com/en-us/data-center/b200/
    ("b200_180gb", "7b"): 11_000,
    ("b200_180gb", "13b"): 6_000,
    ("b200_180gb", "34b"): 2_200,
    ("b200_180gb", "70b"): 1_100,
    # b200_192gb: bandwidth=8000 GB/s  https://www.nvidia.com/en-us/data-center/b200/
    # 7b→14GB: 8000/14×20≈11400→11000
    ("b200_192gb", "7b"): 11_000,
    # 13b→26GB: 8000/26×20≈6150→6000
    ("b200_192gb", "13b"): 6_000,
    # 34b→68GB: 8000/68×20≈2350→2200
    ("b200_192gb", "34b"): 2_200,
    # 70b→140GB: 8000/140×20≈1140→1100
    ("b200_192gb", "70b"): 1_100,
    # b300_228gb: bandwidth=8000 GB/s  https://flopper.io/gpu/nvidia-b300-sxm-288gb/spec-sheet.pdf
    ("b300_228gb", "7b"): 11_000,
    ("b300_228gb", "13b"): 6_000,
    ("b300_228gb", "34b"): 2_200,
    ("b300_228gb", "70b"): 1_100,
    # b300_288gb: RunPod 288GB variant; same die as B300 228GB  https://flopper.io/gpu/nvidia-b300-sxm-288gb/spec-sheet.pdf
    ("b300_288gb", "7b"): 11_000,
    ("b300_288gb", "13b"): 6_000,
    ("b300_288gb", "34b"): 2_200,
    ("b300_288gb", "70b"): 1_100,
    # b300_34gb: B300 MIG slice (34 GB of VRAM); full die bandwidth ~8000 GB/s but
    # MIG partitions share bandwidth — effective ~8000/8 ≈ 1000 GB/s per slice
    # Only 7b fits in 34 GB (fp16: 14 GB model + 20 GB KV cache headroom)
    # 7b→14GB: 1000/14×20≈1430→1400
    ("b300_34gb", "7b"): 1_400,
    # mi300x_192gb: AMD MI300X; bandwidth=5300 GB/s  https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
    # 7b→14GB: 5300/14×20≈7570→7000
    ("mi300x_192gb", "7b"): 7_000,
    # 13b→26GB: 5300/26×20≈4080→4000
    ("mi300x_192gb", "13b"): 4_000,
    # 34b→68GB: 5300/68×20≈1560→1500
    ("mi300x_192gb", "34b"): 1_500,
    # 70b→140GB: 5300/140×20≈757→700
    ("mi300x_192gb", "70b"): 700,
    # rtx_5000_ada_32gb: NVIDIA RTX 5000 Ada; bandwidth=576 GB/s  https://www.nvidia.com/en-us/design-visualization/rtx-5000/
    # 7b→14GB: 576/14×20≈823→800
    ("rtx_5000_ada_32gb", "7b"): 800,
    # 13b→26GB: 576/26×20≈443→400
    ("rtx_5000_ada_32gb", "13b"): 400,
    # rtx_6000_ada_48gb: NVIDIA RTX 6000 Ada; bandwidth=960 GB/s  https://www.nvidia.com/en-us/design-visualization/rtx-6000/
    # 7b→14GB: 960/14×20≈1371→1300
    ("rtx_6000_ada_48gb", "7b"): 1_300,
    # 13b→26GB: 960/26×20≈738→700
    ("rtx_6000_ada_48gb", "13b"): 700,
    # 34b→68GB: 960/68×20≈282→270
    ("rtx_6000_ada_48gb", "34b"): 270,
    # rtx_a5000_24gb: NVIDIA RTX A5000; bandwidth=768 GB/s  https://www.nvidia.com/en-us/design-visualization/rtx-a5000/
    # 7b→14GB: 768/14×20≈1097→1000
    ("rtx_a5000_24gb", "7b"): 1_000,
    # 13b→26GB: 768/26×20≈591→550
    ("rtx_a5000_24gb", "13b"): 550,
    # rtx_a6000_48gb: NVIDIA RTX A6000; bandwidth=768 GB/s  https://www.nvidia.com/en-us/design-visualization/rtx-a6000/
    # 7b→14GB: 768/14×20≈1097→1000
    ("rtx_a6000_48gb", "7b"): 1_000,
    # 13b→26GB: 768/26×20≈591→550
    ("rtx_a6000_48gb", "13b"): 550,
    # 34b→68GB: 768/68×20≈226→220
    ("rtx_a6000_48gb", "34b"): 220,
    # rtx_pro_4000_24gb: NVIDIA RTX PRO 4000 (Blackwell workstation); bandwidth=432 GB/s
    # https://www.nvidia.com/en-us/design-visualization/rtx-pro-4000/
    # 7b→14GB: 432/14×20≈617→600
    ("rtx_pro_4000_24gb", "7b"): 600,
    # 13b→26GB: 432/26×20≈332→300
    ("rtx_pro_4000_24gb", "13b"): 300,
    # rtx_pro_4500_32gb: NVIDIA RTX PRO 4500; bandwidth=576 GB/s
    # https://www.nvidia.com/en-us/design-visualization/rtx-pro-4500/
    # 7b→14GB: 576/14×20≈823→800
    ("rtx_pro_4500_32gb", "7b"): 800,
    # 13b→26GB: 576/26×20≈443→400
    ("rtx_pro_4500_32gb", "13b"): 400,
    # rtx_pro_5000_48gb: NVIDIA RTX PRO 5000; bandwidth=864 GB/s (same die as L40S)
    # https://www.nvidia.com/en-us/design-visualization/rtx-pro-5000/
    # 7b→14GB: 864/14×20≈1234→1200
    ("rtx_pro_5000_48gb", "7b"): 1_200,
    # 13b→26GB: 864/26×20≈665→600
    ("rtx_pro_5000_48gb", "13b"): 600,
    # 34b→68GB: 864/68×20≈254→250
    ("rtx_pro_5000_48gb", "34b"): 250,
    # rtx_pro_6000_96gb: NVIDIA RTX PRO 6000 (Blackwell); bandwidth=1920 GB/s
    # https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000/
    # 7b→14GB: 1920/14×20≈2743→2600
    ("rtx_pro_6000_96gb", "7b"): 2_600,
    # 13b→26GB: 1920/26×20≈1477→1400
    ("rtx_pro_6000_96gb", "13b"): 1_400,
    # 34b→68GB: 1920/68×20≈565→550
    ("rtx_pro_6000_96gb", "34b"): 550,
    # 70b→140GB: 1920/140×20≈274→270
    ("rtx_pro_6000_96gb", "70b"): 270,
    # rtx_pro_6000_48gb: MIG 1/2 slice of RTX PRO 6000 (48 GB); effective BW ≈ 1920/2 = 960 GB/s
    # 7b→14GB: 960/14×20≈1371→1300
    ("rtx_pro_6000_48gb", "7b"): 1_300,
    # 13b→26GB: 960/26×20≈738→700
    ("rtx_pro_6000_48gb", "13b"): 700,
    # 34b→68GB: 960/68×20≈282→270
    ("rtx_pro_6000_48gb", "34b"): 270,
    # rtx_pro_6000_24gb: MIG 1/4 slice of RTX PRO 6000 (24 GB); effective BW ≈ 1920/4 = 480 GB/s
    # Only 7b fits in 24 GB headroom
    # 7b→14GB: 480/14×20≈686→650
    ("rtx_pro_6000_24gb", "7b"): 650,
}


@dataclass
class InfraProvider:
    id: str
    cloud_provider: str
    name: str
    instance_type: str
    gpu_count: int
    gpu_memory_gb: int  # VRAM per GPU; total = gpu_count × gpu_memory_gb
    pricing: InfraProviderPricing
    is_live: bool
    gpu_type: str = ""  # key into _TOKENS_PER_SEC (e.g. "a100_80gb")


# ---------------------------------------------------------------------------
# Hardcoded providers (AWS + GCP)
# ---------------------------------------------------------------------------

_PROVIDERS: list[InfraProvider] = [
    # --- AWS ---
    InfraProvider(
        id="aws_p4d_24xlarge",
        cloud_provider="aws",
        name="AWS p4d.24xlarge",
        instance_type="p4d.24xlarge",
        gpu_count=8,
        gpu_memory_gb=40,
        # 8× A100 40GB, us-east-1
        # on-demand: https://aws.amazon.com/ec2/pricing/on-demand/ (filter: p4d.24xlarge)
        # spot: https://aws.amazon.com/ec2/spot/pricing/ (filter: p4d.24xlarge, us-east-1)
        pricing=InfraProviderPricing(spot=9.83, on_demand=32.77),
        is_live=False,
        gpu_type="a100_40gb",
    ),
    InfraProvider(
        id="aws_p4de_24xlarge",
        cloud_provider="aws",
        name="AWS p4de.24xlarge",
        instance_type="p4de.24xlarge",
        gpu_count=8,
        gpu_memory_gb=80,
        # 8× A100 80GB, us-east-1
        # on-demand: https://aws.amazon.com/ec2/pricing/on-demand/ (filter: p4de.24xlarge)
        # spot: https://aws.amazon.com/ec2/spot/pricing/ (filter: p4de.24xlarge, us-east-1)
        pricing=InfraProviderPricing(spot=13.10, on_demand=40.97),
        is_live=False,
        gpu_type="a100_80gb",
    ),
    InfraProvider(
        id="aws_g6_12xlarge",
        cloud_provider="aws",
        name="AWS g6.12xlarge",
        instance_type="g6.12xlarge",
        gpu_count=4,
        gpu_memory_gb=24,
        # 4× L4 24GB, us-east-1
        # spot: $2.5248 on-demand: $2.6682
        pricing=InfraProviderPricing(spot=2.52, on_demand=2.67),
        is_live=False,
        gpu_type="l4_24gb",
    ),
    InfraProvider(
        id="aws_g6e_12xlarge",
        cloud_provider="aws",
        name="AWS g6e.12xlarge",
        instance_type="g6e.12xlarge",
        gpu_count=4,
        gpu_memory_gb=48,
        # 4× L40S 48GB, us-east-1
        # spot: $3.8622 on-demand: $3.2573 (spot > on-demand is unusual; both listed as-is)
        pricing=InfraProviderPricing(spot=3.86, on_demand=3.26),
        is_live=False,
        gpu_type="l40s_48gb",
    ),
    InfraProvider(
        id="aws_g7e_12xlarge",
        cloud_provider="aws",
        name="AWS g7e.12xlarge",
        instance_type="g7e.12xlarge",
        gpu_count=4,
        gpu_memory_gb=48,
        # 4× L40S 48GB (newer gen), us-east-1
        # spot: $2.8141 on-demand: $3.0366
        pricing=InfraProviderPricing(spot=2.81, on_demand=3.04),
        is_live=False,
        gpu_type="l40s_48gb",
    ),
    InfraProvider(
        id="aws_p5_4xlarge",
        cloud_provider="aws",
        name="AWS p5.4xlarge",
        instance_type="p5.4xlarge",
        gpu_count=2,
        gpu_memory_gb=80,
        # 2× H100 80GB, us-east-1
        # spot: $3.7598 on-demand: N/A (on-demand not publicly listed for p5.4xlarge)
        pricing=InfraProviderPricing(spot=3.76, on_demand=9.50),
        is_live=False,
        gpu_type="h100_80gb",
    ),
    InfraProvider(
        id="aws_p6_b200_48xlarge",
        cloud_provider="aws",
        name="AWS p6-b200.48xlarge",
        instance_type="p6-b200.48xlarge",
        gpu_count=8,
        gpu_memory_gb=192,
        # 8× B200 192GB, us-east-1
        # spot: $23.1961 on-demand: N/A
        pricing=InfraProviderPricing(spot=23.20, on_demand=60.00),
        is_live=False,
        gpu_type="b200_192gb",
    ),
    InfraProvider(
        id="aws_p6_b300_48xlarge",
        cloud_provider="aws",
        name="AWS p6-b300.48xlarge",
        instance_type="p6-b300.48xlarge",
        gpu_count=8,
        gpu_memory_gb=228,
        # 8× B300 228GB, us-east-1
        # spot: $22.4568 on-demand: N/A
        pricing=InfraProviderPricing(spot=22.46, on_demand=58.00),
        is_live=False,
        gpu_type="b300_228gb",
    ),
    # --- GCP ---
    InfraProvider(
        id="gcp_a2_highgpu_1g",
        cloud_provider="gcp",
        name="GCP a2-highgpu-1g",
        instance_type="a2-highgpu-1g",
        gpu_count=1,
        gpu_memory_gb=40,
        # 1× A100 40GB, us-central1
        # https://cloud.google.com/compute/vm-instance-pricing#a2-highgpu (spot + on-demand)
        pricing=InfraProviderPricing(spot=1.10, on_demand=3.67),
        is_live=False,
        gpu_type="a100_40gb",
    ),
    InfraProvider(
        id="gcp_a2_highgpu_2g",
        cloud_provider="gcp",
        name="GCP a2-highgpu-2g",
        instance_type="a2-highgpu-2g",
        gpu_count=2,
        gpu_memory_gb=40,
        # 2× A100 40GB, us-central1
        # https://cloud.google.com/compute/vm-instance-pricing#a2-highgpu (spot + on-demand)
        pricing=InfraProviderPricing(spot=2.20, on_demand=7.35),
        is_live=False,
        gpu_type="a100_40gb",
    ),
    InfraProvider(
        id="gcp_a2_highgpu_4g",
        cloud_provider="gcp",
        name="GCP a2-highgpu-4g",
        instance_type="a2-highgpu-4g",
        gpu_count=4,
        gpu_memory_gb=40,
        # 4× A100 40GB, us-central1
        # https://cloud.google.com/compute/vm-instance-pricing#a2-highgpu (spot + on-demand)
        pricing=InfraProviderPricing(spot=4.40, on_demand=14.69),
        is_live=False,
        gpu_type="a100_40gb",
    ),
    InfraProvider(
        id="gcp_a2_ultragpu_1g",
        cloud_provider="gcp",
        name="GCP a2-ultragpu-1g",
        instance_type="a2-ultragpu-1g",
        gpu_count=1,
        gpu_memory_gb=80,
        # 1× A100 80GB, us-central1
        # https://cloud.google.com/compute/vm-instance-pricing#a2-ultragpu (spot + on-demand)
        pricing=InfraProviderPricing(spot=2.00, on_demand=5.10),
        is_live=False,
        gpu_type="a100_80gb",
    ),
    InfraProvider(
        id="gcp_a2_ultragpu_4g",
        cloud_provider="gcp",
        name="GCP a2-ultragpu-4g",
        instance_type="a2-ultragpu-4g",
        gpu_count=4,
        gpu_memory_gb=80,
        # 4× A100 80GB, us-central1
        # https://cloud.google.com/compute/vm-instance-pricing#a2-ultragpu (spot + on-demand)
        pricing=InfraProviderPricing(spot=8.00, on_demand=20.39),
        is_live=False,
        gpu_type="a100_80gb",
    ),
]


# ---------------------------------------------------------------------------
# Live fetchers — add new providers here
# ---------------------------------------------------------------------------

# Each entry is a callable () -> list[InfraProvider].
# Failures are caught individually; a broken fetcher never blocks the others.
_LIVE_FETCHERS: list[tuple[str, Callable[[], list[InfraProvider]]]] = [
    ("runpod", lambda: _fetch_runpod()),
    # ("vast_ai", lambda: _fetch_vast_ai()),  # TODO: not implemented
]


# Prefill is compute-bound (not memory-bandwidth-bound) and parallelises across the full
# sequence in one forward pass, so it runs ~5× faster than autoregressive decode.
# Rule of thumb from vLLM continuous-batching benchmarks; real values are 3–10× depending
# on sequence length and batch size. Used to amortise input tokens into the time estimate.
_PREFILL_MULTIPLIER = 5


def get_tokens_per_sec(gpu_type: str, model_size_class: str) -> int | None:
    """Look up decode throughput (tokens/sec) for a (gpu_type, model_size_class) pair.

    Returns None when no benchmark data is available.
    """
    return _TOKENS_PER_SEC.get((gpu_type, model_size_class))


def get_prefill_multiplier() -> int:
    """Ratio of prefill TPS to decode TPS. Prefill tokens are divided by this before adding to wall-clock time."""
    return _PREFILL_MULTIPLIER


def _gpu_type_from_display(display_name: str, vram_gb: int) -> str:
    """Derive a gpu_type key from a RunPod GPU display name."""
    name = display_name.lower()
    if "a100" in name:
        return f"a100_{vram_gb}gb"
    if "h200" in name:
        return f"h200_{vram_gb}gb"
    if "h100" in name:
        return f"h100_{vram_gb}gb"
    if "b200" in name:
        return f"b200_{vram_gb}gb"
    if "b300" in name:
        return f"b300_{vram_gb}gb"
    # l40s must be checked before l40 and l4 to avoid partial matches
    if "l40s" in name:
        return f"l40s_{vram_gb}gb"
    if "l40" in name:
        return f"l40_{vram_gb}gb"
    if "l4" in name:
        return f"l4_{vram_gb}gb"
    if "a40" in name:
        return f"a40_{vram_gb}gb"
    if "5090" in name:
        return f"rtx_5090_{vram_gb}gb"
    if "4090" in name:
        return f"rtx_4090_{vram_gb}gb"
    if "3090 ti" in name:
        return f"rtx_3090_ti_{vram_gb}gb"
    if "3090" in name:
        return f"rtx_3090_{vram_gb}gb"
    if "mi300x" in name:
        return f"mi300x_{vram_gb}gb"
    # RTX PRO series (Blackwell workstation) — must be checked before generic Ada checks
    # "pro 6000 mig" matches before "pro 6000" so no special ordering needed within this block
    if "rtx pro 6000" in name or "pro 6000" in name:
        return f"rtx_pro_6000_{vram_gb}gb"
    if "rtx pro 5000" in name:
        return f"rtx_pro_5000_{vram_gb}gb"
    if "rtx pro 4500" in name:
        return f"rtx_pro_4500_{vram_gb}gb"
    if "rtx pro 4000" in name:
        return f"rtx_pro_4000_{vram_gb}gb"
    # Ada Lovelace workstation — check "6000 ada" before "a6000" to avoid collision
    if "rtx 6000 ada" in name:
        return f"rtx_6000_ada_{vram_gb}gb"
    if "rtx 5000 ada" in name:
        return f"rtx_5000_ada_{vram_gb}gb"
    # Ampere workstation
    if "a6000" in name:
        return f"rtx_a6000_{vram_gb}gb"
    if "a5000" in name:
        return f"rtx_a5000_{vram_gb}gb"
    return ""


def _gpu_id(display_name: str, vram_gb: int) -> str:
    slug = display_name.lower().replace(" ", "_")
    return f"runpod_{slug}_{vram_gb}gb"


def _fetch_runpod() -> list[InfraProvider]:
    query = "{ gpuTypes { displayName memoryInGb securePrice communityPrice } }"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post("https://api.runpod.io/graphql", json={"query": query})
        resp.raise_for_status()

    gpu_types = resp.json().get("data", {}).get("gpuTypes", [])

    results: list[InfraProvider] = []
    for g in gpu_types:
        spot = g.get("communityPrice") or None
        on_demand = g.get("securePrice") or g.get("communityPrice")
        if not on_demand:
            continue
        vram = g["memoryInGb"]
        if vram < _MIN_TOTAL_VRAM_GB:
            continue
        display = g["displayName"]
        results.append(
            InfraProvider(
                id=_gpu_id(display, vram),
                cloud_provider="runpod",
                name=f"RunPod {display}",
                instance_type=f"{display} {vram}GB",
                gpu_count=1,
                gpu_memory_gb=vram,
                pricing=InfraProviderPricing(
                    spot=float(spot) if spot else None,
                    on_demand=float(on_demand),
                ),
                is_live=True,
                gpu_type=_gpu_type_from_display(display, vram),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_static_infra_provider_ids() -> set[str]:
    """Return IDs of hardcoded (non-live) infra providers. Safe to call synchronously."""
    return {p.id for p in _PROVIDERS}


async def fetch_infra_providers() -> list[InfraProvider]:
    """Fetch live prices from all registered fetchers and combine with hardcoded providers."""
    live: list[InfraProvider] = []
    for name, fetcher in _LIVE_FETCHERS:
        try:
            live.extend(fetcher())
        except Exception as exc:
            logger.warning("Failed to fetch prices from %s: %s", name, exc)
    return live + _PROVIDERS


def _provider_to_record(p: InfraProvider, fetched_at: str) -> InfraProviderRecord:
    return InfraProviderRecord(
        id=p.id,
        cloud_provider=p.cloud_provider,
        name=p.name,
        instance_type=p.instance_type,
        gpu_count=p.gpu_count,
        gpu_memory_gb=p.gpu_memory_gb,
        spot_price=p.pricing.spot,
        on_demand_price=p.pricing.on_demand,
        is_live=p.is_live,
        gpu_type=p.gpu_type,
        fetched_at=fetched_at,
    )


def _record_to_provider(r: InfraProviderRecord) -> InfraProvider:
    return InfraProvider(
        id=r.id,
        cloud_provider=r.cloud_provider,
        name=r.name,
        instance_type=r.instance_type,
        gpu_count=r.gpu_count,
        gpu_memory_gb=r.gpu_memory_gb,
        pricing=InfraProviderPricing(spot=r.spot_price, on_demand=r.on_demand_price),
        is_live=r.is_live,
        gpu_type=r.gpu_type,
    )


def _load_providers_from_db() -> list[InfraProvider] | None:
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(select(InfraProviderRecord)).all()
        if not rows:
            return None
        return [_record_to_provider(r) for r in rows]


def _persist_providers(providers: list[InfraProvider]) -> None:
    fetched_at = datetime.now(UTC).isoformat()
    with get_session() as session:
        for p in providers:
            session.merge(_provider_to_record(p, fetched_at))
        session.commit()


def get_infra_providers_from_db() -> list[InfraProvider]:
    """Return providers from DB; raises if DB is empty (run refresh_infra_providers first)."""
    stored = _load_providers_from_db()
    if stored is None:
        raise RuntimeError("Infra provider data not found in DB. Run `refresh_infra_providers()` to seed it.")
    return stored


async def get_infra_providers() -> list[InfraProvider]:
    """Return providers from DB; fetch live and persist if DB is empty."""
    stored = _load_providers_from_db()
    if stored is not None:
        return stored

    providers = await fetch_infra_providers()
    _persist_providers(providers)
    return providers


async def refresh_infra_providers() -> list[InfraProvider]:
    """Force a live fetch, persist to DB, and return fresh data."""
    providers = await fetch_infra_providers()
    _persist_providers(providers)
    return providers
