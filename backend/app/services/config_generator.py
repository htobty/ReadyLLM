"""确定性配置生成器

根据硬件规格 + 模型文件信息，用纯计算逻辑生成一组"必然正确"的基础参数。
不依赖 LLM，不需要种子，对新用户通用。

每条规则都有明确的计算依据：
  1. 模型权重 < 显存×0.85 → ngl=all（全量 GPU 卸载）
  2. 剩余显存决定 KV cache 量化级别（优先高精度，放不下才降级）
  3. 模型支持 MTP → 启用投机解码 + draft 层全进 GPU
  4. 剩余显存算 batch 上限
  5. 线程数 = CPU 物理核数（不超过 32）

设计原则：
  - 宁可保守也不 OOM：所有估算留 15% 余量
  - 每条规则可独立单测
  - 输出可直接作为 llama-server 启动参数
"""

import re
from typing import Optional

# KV cache 每 token 每层的显存占用（字节），按量化级别
# 近似公式：2(K+V) × head_dim × bytes_per_element
# 对 27B 级模型（head_dim≈128, 64层）：每 token 约 2×128×64×bytes = 16384×bytes
_KV_BYTES_PER_TOKEN_PER_LAYER = {
    "f16": 2.0,    # 2 bytes per element
    "q8_0": 1.0,   # 1 byte
    "q4_0": 0.5,   # 0.5 byte
}

# 投机解码 draft 模型的额外显存开销估算（GB）
# MTP draft 通常是主模型的 1-2 层，约 0.3-0.8GB
_DRAFT_OVERHEAD_GB = 0.5

# 显存安全余量（留给 CUDA context、碎片、激活值）
_VRAM_HEADROOM = 0.85  # 只用 85% 显存


def generate_config(
    gpu_vram_gb: float,
    model_size_gb: float,
    model_filename: str,
    ctx_size: int,
    cpu_cores: int = 8,
    cpu_threads: int = 16,
    num_layers: int = 0,
) -> dict:
    """生成确定性基础配置。

    Args:
        gpu_vram_gb: GPU 显存总量（GB）
        model_size_gb: 模型文件大小（GB）
        model_filename: 模型文件名（用于推断量化类型和 MTP 支持）
        ctx_size: 用户要求的上下文长度
        cpu_cores: CPU 物理核数
        cpu_threads: CPU 逻辑线程数
        num_layers: 模型层数（0 则自动推断）

    Returns:
        {
            "params": {...},       # llama-server 参数
            "reasoning": [...],    # 每条规则的决策理由（供 AI 和用户查看）
            "warnings": [...],     # 潜在风险警告
        }
    """
    reasoning = []
    warnings = []
    params = {}

    # ========== 规则 0：推断模型层数 ==========
    if num_layers <= 0:
        num_layers = _infer_layers(model_filename, model_size_gb)
        reasoning.append(f"推断模型层数: {num_layers}（基于文件名和大小）")

    # ========== 规则 1：GPU 卸载策略 ==========
    usable_vram = gpu_vram_gb * _VRAM_HEADROOM
    model_fits = model_size_gb <= usable_vram

    if model_fits:
        params["n-gpu-layers"] = "all"
        reasoning.append(
            f"模型 {model_size_gb:.1f}GB < 可用显存 {usable_vram:.1f}GB "
            f"({gpu_vram_gb}×{_VRAM_HEADROOM}) → 全量 GPU 卸载"
        )
    else:
        # 模型放不下：计算能放多少层
        layers_fit = int((usable_vram / model_size_gb) * num_layers * 0.9)
        params["n-gpu-layers"] = str(max(layers_fit, 1))
        warnings.append(
            f"模型 {model_size_gb:.1f}GB 超过可用显存 {usable_vram:.1f}GB，"
            f"只能卸载 {layers_fit}/{num_layers} 层到 GPU，性能会显著下降"
        )
        reasoning.append(
            f"模型放不下 → 部分卸载 {layers_fit} 层（这是唯一允许非 all 的情况）"
        )

    # ========== 规则 2+3：KV cache 量化 + 投机解码（联合决策） ==========
    # 核心原则：投机解码的收益（+50~100%）远大于 cache 精度差异（<5%），
    # 所以优先保证投机解码，必要时降 cache 量化来腾显存。
    remaining_vram = usable_vram - model_size_gb
    supports_mtp = _supports_mtp(model_filename)

    if supports_mtp and model_fits:
        # 尝试从高到低找"能同时容纳 KV + draft + batch 余量"的 cache 级别
        cache_type = None
        for ct in ["f16", "q8_0", "q4_0"]:
            kv_gb = _estimate_kv_gb(ctx_size, num_layers, ct)
            after = remaining_vram - kv_gb - _DRAFT_OVERHEAD_GB
            if after > 1.0:  # 至少留 1GB 给 batch/激活
                cache_type = ct
                break
        if cache_type is None:
            # 即使 q4_0 也放不下 draft → 放弃投机，用最高精度 cache
            cache_type = _choose_cache_type(remaining_vram, ctx_size, num_layers)
            warnings.append("显存余量不足以同时容纳 draft 模型，跳过投机解码")
            reasoning.append("显存不足以启用投机解码，回退到无投机方案")
        else:
            params["spec-type"] = "draft-mtp"
            params["spec-draft-n-max"] = "3"
            params["gpu-layers-draft"] = "all"
            params["spec-draft-ngl"] = "all"
            kv_usage = _estimate_kv_gb(ctx_size, num_layers, cache_type)
            reasoning.append(
                f"模型支持 MTP，为保证投机解码选用 cache={cache_type}"
                f"（KV 占 {kv_usage:.1f}GB + draft {_DRAFT_OVERHEAD_GB}GB，"
                f"剩余 {remaining_vram - kv_usage - _DRAFT_OVERHEAD_GB:.1f}GB）"
            )
    else:
        # 不支持 MTP 或模型放不下 → 只选最高精度 cache
        cache_type = _choose_cache_type(remaining_vram, ctx_size, num_layers)
        kv_usage = _estimate_kv_gb(ctx_size, num_layers, cache_type)
        if not supports_mtp:
            reasoning.append("模型不支持 MTP 投机解码（文件名未检测到相关标记）")

    params["cache-type-k"] = cache_type
    params["cache-type-v"] = cache_type
    kv_usage = _estimate_kv_gb(ctx_size, num_layers, cache_type)
    reasoning.append(
        f"剩余显存 {remaining_vram:.1f}GB，ctx={ctx_size}，"
        f"KV cache({cache_type}) 约占 {kv_usage:.1f}GB"
    )

    # ========== 规则 4：batch 大小 ==========
    # batch 主要影响预填充速度，解码阶段影响小
    # 经验：显存余量 > 4GB 时用 4096，> 2GB 用 2048，否则 1024
    after_all = remaining_vram - kv_usage
    if supports_mtp and model_fits:
        after_all -= _DRAFT_OVERHEAD_GB

    if after_all > 4.0:
        batch, ubatch = 4096, 1024
    elif after_all > 2.0:
        batch, ubatch = 2048, 512
    else:
        batch, ubatch = 1024, 256
    params["batch-size"] = str(batch)
    params["ubatch-size"] = str(ubatch)
    reasoning.append(
        f"扣除模型+KV+draft后剩余 {after_all:.1f}GB → batch={batch}, ubatch={ubatch}"
    )

    # ========== 规则 5：线程数 ==========
    # 线程数 = 物理核数，但不超过 32（超过后收益递减）
    threads = min(cpu_cores, 32)
    params["threads"] = str(threads)
    reasoning.append(f"CPU {cpu_cores} 核 → threads={threads}")

    # ========== 固定参数 ==========
    params["flash-attn"] = "on"
    params["fit"] = "off"
    params["ctx-size"] = str(ctx_size)

    return {
        "params": params,
        "reasoning": reasoning,
        "warnings": warnings,
    }


# ==================== 内部计算函数 ====================

def _infer_layers(filename: str, size_gb: float) -> int:
    """从文件名和大小推断模型层数"""
    # 常见模型参数量 → 层数映射
    name_lower = filename.lower()
    if "70b" in name_lower:
        return 80
    elif "32b" in name_lower or "34b" in name_lower:
        return 64
    elif "27b" in name_lower:
        return 64
    elif "14b" in name_lower:
        return 40
    elif "8b" in name_lower or "7b" in name_lower:
        return 32
    elif "3b" in name_lower or "4b" in name_lower:
        return 36
    elif "1b" in name_lower or "0.5b" in name_lower or "0.6b" in name_lower:
        return 24
    # 按大小粗估：每层约 0.2-0.4GB（取决于量化）
    return max(int(size_gb / 0.25), 16)


def _supports_mtp(filename: str) -> bool:
    """判断模型是否支持 MTP 投机解码。
    Qwen3 系列（含 3.5/3.6/3.7/3.8）原生支持 MTP。
    其他模型需要文件名中有明确标记。"""
    name_lower = filename.lower()
    # Qwen3 系列全部支持 MTP
    if re.search(r"qwen3[\.\-]?\d", name_lower):
        return True
    if "qwen3" in name_lower:
        return True
    # 显式标记
    if "mtp" in name_lower:
        return True
    return False


def _estimate_kv_gb(ctx_size: int, num_layers: int, cache_type: str) -> float:
    """估算 KV cache 显存占用（GB）
    公式：2(K+V) × ctx × head_dim × layers × bytes_per_element
    head_dim 取 128（主流模型）"""
    head_dim = 128
    bytes_per_elem = _KV_BYTES_PER_TOKEN_PER_LAYER.get(cache_type, 2.0)
    total_bytes = 2 * ctx_size * head_dim * num_layers * bytes_per_elem
    return total_bytes / (1024 ** 3)


def _choose_cache_type(remaining_vram: float, ctx_size: int, num_layers: int) -> str:
    """根据剩余显存选择最高精度的 cache 量化级别。
    优先 f16（最快），放不下降 q8_0，再放不下降 q4_0。"""
    for cache_type in ["f16", "q8_0", "q4_0"]:
        kv_gb = _estimate_kv_gb(ctx_size, num_layers, cache_type)
        if kv_gb <= remaining_vram * 0.8:  # 留 20% 余量给 batch
            return cache_type
    # 都放不下，用最小的
    return "q4_0"
