"""智能调优压测服务（两阶段搜索版）

在用户当前硬件配置下，为目标机上的某个模型搜索最优推理参数。

设计要点：
  1. 参数分三类：
     - 约束类（用户/硬件定死）：模型、端口、ctx 下限 → 不搜，作硬约束
     - 离散高影响类：spec-type / cache-type-k,v / n-gpu-layers → coarse 阶段搜
     - 连续微调类：batch-size / ubatch-size / spec-draft-n-max → fine 阶段坐标下降
  2. 显存可行性预检：估算权重+KV+cache 占用，放不下的组合直接跳过，不浪费启动时间
  3. 两阶段搜索：coarse 定主导因素 → fine 在最优组合附近收敛
  4. 可信测速：预热 2 次 + 正式 3 次取中位数（解码测速用 ignore_eos 跑满固定窗口，
     避免模型提前 EOS 导致测量窗口塌缩），记录解码/预填充/TTFT/GPU 利用率
  5. 基线对比：以用户原始参数为 baseline 先测，输出"推荐 vs 当前"

优化目标可选：latency(端到端体感,默认) / throughput(纯解码吞吐) / prefill(长文本预填充)。
无引擎/无模型时明确报错，绝不返回模拟数据。
"""

import os
import re
import shlex
import threading
import time
import json
import uuid
from dataclasses import dataclass
from statistics import median
from typing import Optional, List, Dict

from .executor import Executor
from .engine_adapter import StartParams
from .llama_cpp import LlamaCppAdapter
from .i18n import L, get_lang, set_lang
from ..models.target import Target, get_target

_JOBS: dict = {}
_LOCK = threading.Lock()

# 基准测试 prompt（固定，保证各组可比）
_BENCH_PROMPT = "请用一句话解释什么是大语言模型。"
# 长 prompt：约 2000+ tokens，足以触发多轮 batch 拆分，测出预填充真实瓶颈
_BENCH_LONG_PROMPT = (
    "Transformer架构是现代大语言模型的基础。其核心组件包括多头自注意力机制、"
    "位置编码、前馈神经网络和层归一化。自注意力机制允许模型在处理每个词时关注"
    "输入序列中的所有其他位置，从而捕获长距离依赖关系。多头注意力将表示空间投影"
    "到多个子空间中并行计算注意力，增强了模型的表达能力。位置编码使用正弦和余弦"
    "函数为序列中的每个位置生成唯一的向量表示，使模型能够感知词的顺序信息。"
    "前馈网络对每个位置独立应用两层全连接变换，引入非线性特征提取能力。"
    "层归一化和残差连接则确保深层网络的训练稳定性。在推理阶段，KV缓存机制"
    "避免了重复计算已处理位置的键值对，显著提升了自回归生成的效率。"
    "投机解码技术通过小型草稿模型预测多个候选token，再由大模型并行验证，"
    "从而在不损失质量的前提下加速生成过程。量化技术通过降低权重和激活值的"
    "数值精度来减少显存占用和计算量，使得更大的模型能够在消费级硬件上运行。"
    "常见的量化方案包括GPTQ、AWQ、GGML格式的各种量化级别如Q4_0、Q4_K_M、"
    "Q5_K_M、Q8_0等，它们在精度损失和压缩率之间提供了不同的权衡选择。"
    "Flash Attention算法通过分块计算和在线softmax技巧，将注意力计算的显存复杂度"
    "从二次方降低到线性，使得处理超长上下文成为可能。PagedAttention则将KV缓存"
    "组织成类似操作系统虚拟内存的页表结构，支持高效的显存管理和多请求并发。"
    "在部署层面，模型并行策略包括张量并行、流水线并行和序列并行，分别适用于"
    "不同的硬件拓扑和模型规模。推理引擎如llama.cpp、vLLM、TensorRT-LLM等"
    "各自针对不同的硬件平台和优化目标进行了深度优化。连续批处理技术允许动态"
    "地将新请求插入正在处理的批次中，提高了GPU的利用率和系统吞吐量。"
    "推测性解码的变体包括Medusa、EAGLE和Lookahead Decoding，它们通过不同的"
    "草稿生成策略在速度和质量之间取得平衡。模型蒸馏和剪枝技术则从模型结构层面"
    "减少计算需求，知识蒸馏让小模型学习大模型的输出分布，结构化剪枝移除冗余的"
    "注意力头和神经元。混合专家模型通过门控网络将输入路由到少数专家子网络，"
    "在保持参数量的同时大幅降低每次前向传播的计算量。这些技术的组合使用使得"
    "在单张消费级显卡上部署数十亿参数的模型成为现实，为本地化AI应用奠定了基础。"
) * 6  # ×6 ≈ 2400+ tokens
# 解码测速的固定生成窗口。必须配合 ignore_eos 把窗口跑满：基准 prompt 是「请用一句话
# 解释什么是大语言模型」，不加 ignore_eos 时模型约 24 个 token 就 EOS，n_predict 形同虚设。
# 而 llama.cpp 的 predicted_per_second 分母从第二个 token 才开始计时（实测恒等于
# (predicted_n-1)/predicted_ms），窗口越短越吃亏。同机同参数实测三档：
#   24 token（改动前）49~56 t/s → 256 token 85.2 t/s → 512 token 98.6 t/s
# 512 档已与"强制跑 512 token 长输出"的真值 99.7 t/s 重合，故取 512：
# 单次多花约 5 秒，远小于窗口过短把吞吐低估近一半的代价。
_BENCH_MAX_TOKENS = 512
_BENCH_REPEATS = 3   # 正式测速重复次数，取中位数
_BENCH_WARMUP = 2    # 预热次数，丢弃不计分：模型刚加载完的首个请求比稳态低约 18%

# ==================== 参数分层 ====================

# 离散高影响参数：coarse 阶段搜索
SPEC_OPTIONS = ["off", "draft-mtp"]           # 投机解码方式
CACHE_OPTIONS = ["f16", "q8_0", "q4_0"]       # KV cache 量化（越省显存越大 ctx）
NGL_OPTIONS = ["all", "0"]                    # GPU 卸载层数（all 全进 GPU；0 全 CPU 兜底）

# 连续微调参数：fine 阶段坐标下降
CONTINUOUS_GRID = {
    "batch-size": [1024, 2048, 4096, 8192],
    "ubatch-size": [128, 256, 512, 1024],
    "threads": [16, 24, 32],            # CPU 线程，影响预填充与 CPU 端协同
    "spec-draft-n-max": [2, 3, 4, 5],   # 投机一次预测多少 token
    "spec-draft-n-min": [1, 2, 3],      # 投机最少接受阈值，影响投机效率
    # 投机解码的接受概率下限：draft 头一旦不够确定就提前收手，不再往下猜。
    # llama.cpp 默认 0.0（不设限）；调高可让投机在开放文本上不浪费算力，
    # 使更大的 spec-draft-n-max 收益更稳（见 llama.cpp discussion #25198）
    "spec-draft-p-min": [0.0, 0.5, 0.8],
}

# 目标可选评分权重：(解码速度, 预填充速度, TTFT)
GOAL_WEIGHTS = {
    "latency":    {"decode": 0.5, "prefill": 0.3, "ttft": 0.2},
    "throughput": {"decode": 1.0, "prefill": 0.0, "ttft": 0.0},
    "prefill":    {"decode": 0.2, "prefill": 0.8, "ttft": 0.0},
}
GOAL_KEYS = {
    "latency": "tune.goal.latency",
    "throughput": "tune.goal.throughput",
    "prefill": "tune.goal.prefill",
}


def goal_label(goal: str) -> str:
    """按当前界面语言取优化目标名称（模块级常量不能存 L() 结果，须运行期求值）"""
    return L(GOAL_KEYS.get(goal, goal))


def _fmt_num(val) -> str:
    """数值参数转命令行字符串：1.0 → "1"、0.8 → "0.8"（整数不带小数点，贴近手写习惯）"""
    if isinstance(val, str):
        return val
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    return str(int(f)) if f == int(f) else str(f)


def _parse_num(val) -> float:
    """命令行参数值转 float，兼容 "0.8" / "2" / 2 等写法；非法值返回 0.0"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _normalize_cfg(spec_type, cache_type, ngl, batch, ubatch, draft_n_max,
                   draft_p_min=0.0) -> dict:
    """一个完整配置 = 离散主导因素 + 连续微调参数

    draft_p_min 默认 0.0，与 llama.cpp 的 --spec-draft-p-min 默认值保持一致：
    coarse 阶段按默认值起跑，交给 fine 阶段的坐标下降去搜更优的概率下限。
    """
    cfg = {
        "spec-type": spec_type,
        "cache-type-k": cache_type,
        "cache-type-v": cache_type,
        "n-gpu-layers": ngl,
        "batch-size": str(batch),
        "ubatch-size": str(ubatch),
        "threads": "24",
    }
    if spec_type != "off":
        cfg["spec-draft-n-max"] = str(draft_n_max)
        cfg["spec-draft-n-min"] = "2"
        cfg["spec-draft-p-min"] = _fmt_num(draft_p_min)
    return cfg


def _cfg_label(cfg: dict) -> str:
    return " / ".join(f"{k}={_fmt_num(v)}" for k, v in cfg.items())


def _args_list(cfg: dict, target: Target, ctx_size: int) -> List[str]:
    """把配置 dict 转成 llama-server 命令行参数列表（含固定项与 ctx）"""
    args = []
    for k, v in cfg.items():
        if k == "n-gpu-layers":
            # llama-server 同时认 --n-gpu-layers 与 --gpu-layers，用标准名
            args += ["--n-gpu-layers", str(v)]
        else:
            args += [f"--{k}", str(v)]
    args += [
        "--ctx-size", str(ctx_size),
        "--flash-attn", "on",
        # 关键：必须显式关闭 fit。fit 默认 on，会"自作主张"下调我们设的 batch/ubatch
        # 以塞进它认为安全的显存余量，导致搜索时实际生效参数 ≠ 我们测的参数，结果失真。
        "--fit", "off",
        "--metrics",
        "--host", "0.0.0.0",
        "--port", str(target.service_port),
    ]
    return args


# ==================== 显存可行性预检 ====================

# KV cache 量化每元素字节数
_CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

_GGUF_META_SCRIPT = (
    "import struct,sys\n"
    "p=sys.argv[1]\n"
    "try:\n"
    " f=open(p,'rb'); d=f.read(1<<20); f.close()\n"
    " if len(d)<24 or d[:4]!=b'GGUF': print('NONE'); sys.exit(0)\n"
    " nkv=struct.unpack_from('<Q',d,16)[0]; off=24\n"
    " bt=bd=bl=None\n"
    " def cstr(o):\n"
    "  e=d.index(b'\\x00',o); return d[o:e].decode('utf-8','replace'),e+1\n"
    " for i in range(nkv):\n"
    "  k,off=cstr(off); vt=struct.unpack_from('<I',d,off)[0]; off+=4\n"
    "  if vt==4:\n"
    "   v,off=cstr(off)\n"
    "   if k=='general.architecture': arch=v\n"
    "  elif vt==1:\n"
    "   v=struct.unpack_from('<q',d,off)[0]; off+=8\n"
    "   if k=='general.n_layer': bl=v\n"
    "  elif vt==0:\n"
    "   v=struct.unpack_from('<i',d,off)[0]; off+=4\n"
    "   if k=='general.hidden_size': hid=v\n"
    "   elif k=='attention.head_count': hc=v\n"
    "   elif k=='attention.kv_head_count': khc=v\n"
    "  elif vt==2:\n"
    "   v=struct.unpack_from('<f',d,off)[0]; off+=4\n"
    "  elif vt==5:\n"
    "   et=struct.unpack_from('<I',d,off)[0]; off+=4\n"
    "   n=struct.unpack_from('<Q',d,off)[0]; off+=8\n"
    "   if k=='attention.pattern' and et==1:\n"
    "    bt=[struct.unpack_from('<q',d,off+j*8)[0] for j in range(n)]; off+=n*8\n"
    "   else:\n"
    "    ok=True\n"
    "    for _ in range(n):\n"
    "     if et==4:\n"
    "      _,off=cstr(off)\n"
    "     elif et==1: off+=8\n"
    "     elif et==2: off+=4\n"
    "     elif et in (0,6,8): off+=4\n"
    "     elif et in (7,9): off+=2\n"
    "     elif et==3: off+=1\n"
    "     else: ok=False; break\n"
    "    if not ok: break\n"
    "   continue\n"
    "  else:\n"
    "   if vt==3: off+=1\n"
    "   elif vt in (6,8): off+=1\n"
    "   elif vt in (7,9): off+=2\n"
    "   else: break\n"
    " print(f'ARCH={arch} BLOCKS={bl} PATTERN={bt} HIDDEN={hid} HC={hc} KHC={khc}')\n"
    "except Exception:\n"
    " print('NONE')\n"
)


def _read_gguf_meta(executor: Executor, target: Target, model_path: str) -> dict:
    """读 GGUF 文件头里的模型结构元数据（不依赖 llama-gguf 工具，直接解析文件）。

    返回 {"arch","blocks","pattern"}；读不到（非 GGUF / 命令失败）返回 {}。
    混合注意力（如 Qwen 系列）的 attention.pattern 里 0=滑窗层（不存逐 token
    KV）、-1=全注意力层——KV 缓存只占全注意力层的份额，硬编码 64 层会把
    这类模型的显存高估 3~4 倍（这正是之前"全组合显存不足"误判的根源）。
    """
    if target.os == "windows":
        return {}
    script_path = "/tmp/readyllm_gguf_meta.py"
    if not executor.write_file(_GGUF_META_SCRIPT, script_path):
        return {}
    r = executor.run(f"python3 {script_path} {json.dumps(model_path)}", timeout=25)
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
    if not line.startswith("ARCH="):
        return {}
    out = {}
    for part in line[5:].split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k == "PATTERN":
            try:
                out["pattern"] = [int(x) for x in v.split(",") if x]
            except ValueError:
                pass
        elif k == "BLOCKS":
            try:
                out["blocks"] = int(v)
            except ValueError:
                pass
        elif k == "ARCH":
            out["arch"] = v
        elif k == "HIDDEN":
            try:
                out["hidden"] = int(v)
            except ValueError:
                pass
        elif k == "HC":
            try:
                out["head_count"] = int(v)
            except ValueError:
                pass
        elif k == "KHC":
            try:
                out["kv_heads"] = int(v)
            except ValueError:
                pass
    return out


def _kv_layers_from_meta(meta: dict) -> int:
    """从元数据算"真正存逐 token KV 的层数"；缺元数据时回退 64（旧经验值）"""
    pattern = meta.get("pattern")
    if pattern:
        n = 0
        for v in pattern:
            if v == -1:          # 全注意力层：每 token 都存 KV
                n += 1
            elif isinstance(v, int) and 1 <= v <= 8:
                n += 1           # 滑窗层：只存最后 v 个 token 的 KV，近似按 1 层计
        if n:
            return n
    blocks = meta.get("blocks")
    if blocks and blocks > 0:
        return blocks
    return 64


@dataclass
class _VRAMInfo:
    """显存预检的全部依据：模型大小 + 显卡显存 + 模型结构元数据。

    两种可靠来源：
    1) meta 模式：GGUF 头里的层数/头数（标准 GGUF 才读得到）；
    2) measured 模式：基线跑起来后，用真实占用的显存反推"每 token KV 字节数"
       （measured_at_bytes > 0 表示处于此模式）——对非标准 GGUF 布局的
       自定义模型文件，这是唯一可靠的来源。
    """
    model_size_gb: float = 0.0
    gpu_vram_gb: float = 0.0
    kv_layers: int = 64
    kv_heads: int = 0      # 0 = 元数据里没有，按保守估算
    kv_head_dim: int = 0
    bytes_per_kv_token: int = 0   # 全精度(f16)下每 token 每层的 KV 字节数
    measured_at_bytes: float = 0.0  # measured 模式：实测时用的 KV 量化字节数

    @property
    def meta_ok(self) -> bool:
        return self.bytes_per_kv_token > 0


def _estimate_vram_gb(vram: "_VRAMInfo", ctx_size: int, cache_type: str) -> float:
    """按模型真实结构估算 GPU 全量占用 GB：权重 + KV cache。

    meta 模式：KV = 2(K+V) × ctx × kv层数 × kv头数 × 头维度 × 量化字节数。
    measured 模式：KV = ctx × 实测每token字节数 ×（本组量化/实测量化）。
    两者都不可用时退回旧经验公式（kv 维度 8192 × 64 层）。
    """
    cache_b = _CACHE_BYTES.get(cache_type, 2.0)
    if vram.meta_ok:
        if vram.measured_at_bytes > 0:
            kv_gb = (ctx_size * vram.kv_layers * vram.bytes_per_kv_token
                     * (cache_b / vram.measured_at_bytes)) / (1024 ** 3)
        else:
            kv_per_layer = vram.bytes_per_kv_token * cache_b
            kv_gb = 2 * ctx_size * vram.kv_layers * kv_per_layer / (1024 ** 3)
    else:
        kv_bytes = 2 * ctx_size * 8192 * cache_b
        kv_gb = kv_bytes * 64 / (1024 ** 3)
    return vram.model_size_gb + kv_gb


def _fits_vram(cfg: dict, vram: "_VRAMInfo", ctx_size: int) -> bool:
    """判断配置是否放得进显存；n-gpu-layers=0 视为 CPU 兜底，总能'放下'（慢）。

    数字型 n-gpu-layers 按"每层权重均摊"估部分卸载的占用；
    留 10% 余量给激活值/显存碎片。
    完全估算不了（元数据读不到且没有实测校准）时不拦截、交给真实启动去验证——
    错误的猜测式拦截（旧版硬编码 64 层）比放行更糟。
    """
    if cfg.get("n-gpu-layers") == "0":
        return True
    if not vram.gpu_vram_gb:
        # 读不到显存信息：不拦截，交给真实启动去验证
        return True
    if not vram.meta_ok:
        # 估算不可靠：不拦截（旧版在这里用错误公式拦掉了能跑的组合）
        return True
    ngl = cfg.get("n-gpu-layers", "all")
    if ngl == "all":
        est = _estimate_vram_gb(vram, ctx_size, cfg.get("cache-type-k", "f16"))
        return est <= vram.gpu_vram_gb * 0.9
    # 部分卸载：权重按层数均摊，KV 只算进 GPU 的那些层
    try:
        ngl_n = int(str(ngl))
    except ValueError:
        est = _estimate_vram_gb(vram, ctx_size, cfg.get("cache-type-k", "f16"))
        return est <= vram.gpu_vram_gb * 0.9
    if ngl_n <= 0:
        return True
    # measured 模式没有真实层数，按 64 层近似均摊
    layers_for_split = 64 if vram.kv_layers <= 1 else vram.kv_layers
    per_layer_gb = vram.model_size_gb / layers_for_split
    weights_gpu = per_layer_gb * ngl_n
    est_all = _estimate_vram_gb(vram, ctx_size, cfg.get("cache-type-k", "f16"))
    kv_all = est_all - vram.model_size_gb
    kv_gpu = kv_all * min(ngl_n, layers_for_split) / layers_for_split
    return (weights_gpu + kv_gpu) <= vram.gpu_vram_gb * 0.9


def _calibrate_vram(vram: "_VRAMInfo", metrics: dict, baseline_cfg: dict,
                    ctx_size: int) -> bool:
    """基线跑通后，用真实显存占用反推"每 token KV 字节数"。

    标准 GGUF 读得到结构元数据时不需要这一步；对自定义量化管线产出的
    非标准 GGUF 文件，实测是唯一可靠的估算来源：
        KV每token字节 ≈ (基线占用显存 − 模型权重 − 1.5GB 激活/碎片) / ctx
    成功返回 True。
    """
    if vram.meta_ok or not vram.gpu_vram_gb:
        return False
    used_gb = vram.gpu_vram_gb * metrics.get("gpu_mem_pct", 0) / 100.0
    kv_gb = used_gb - vram.model_size_gb - 1.5
    if kv_gb <= 0.2:
        return False
    kv_bytes_per_token = kv_gb * (1024 ** 3) / max(ctx_size, 1)
    if kv_bytes_per_token < 16:
        return False
    vram.kv_layers = 1
    vram.kv_heads = 1
    vram.kv_head_dim = 1
    vram.bytes_per_kv_token = int(kv_bytes_per_token)
    vram.measured_at_bytes = _CACHE_BYTES.get(
        baseline_cfg.get("cache-type-k", "f16"), 2.0)
    return True


def _read_gpu_mem_free_gb(executor: Executor, target: Target) -> float:
    """当前 GPU 空闲显存 GB；读不到返回 0.0"""
    r = executor.run(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", timeout=8)
    vals = [float(x) for x in r.stdout.split() if x.strip().isdigit()]
    if vals:
        return round(sum(vals) / 1024, 1)
    if target.os == "linux":
        r = executor.run(
            "for d in /sys/class/drm/card*/device; do "
            "t=$(cat $d/mem_info_vram_total 2>/dev/null); "
            "u=$(cat $d/mem_info_vram_used 2>/dev/null); "
            "[ -n \"$t\" ] && echo $((t - ${u:-0})); done", timeout=8)
        best = 0.0
        for line in r.stdout.split():
            try:
                best = max(best, int(line) / 1024 ** 3)
            except ValueError:
                continue
        return round(best, 1)
    return 0.0


def _wait_vram_freed(executor: Executor, target: Target, job_id: str,
                     used_before_gb: float, deadline_s: int = 90) -> None:
    """停服后轮询直到显存真正释放，再放行下一组测试。

    固定 sleep 2 秒不够：大模型显存归还可能要几十秒，抢跑会在"显存已满"
    的 GPU 上启动下一组，表现成莫名其妙超时（Issue #1 run 2 的 4 次超时即此）。
    """
    if used_before_gb <= 0:
        return
    target_free = max(used_before_gb - 0.5, 0.5)
    deadline = time.time() + deadline_s
    warned = False
    while time.time() < deadline:
        free = _read_gpu_mem_free_gb(executor, target)
        if free >= target_free:
            return
        if not warned:
            _append_log(job_id, L("tune.log.vram_wait"))
            warned = True
        time.sleep(3)
    if warned:
        _append_log(job_id, L("tune.log.vram_still_busy"))


def _ensure_stopped(executor: Executor, target: Target, engine: "LlamaCppAdapter",
                    job_id: str) -> None:
    """停掉 llama-server 并确认它真的退出了、显存释放了，再返回。

    有进程管理器（如 llama-swap）时客户端请求可能随时把模型拉起来；
    旧进程没退就启动下一组，两个服务抢端口/抢显存，测速全部失真。
    """
    used_before = _gpu_used_before(executor, target)
    for attempt in range(3):
        engine.stop()
        deadline = time.time() + 15
        while time.time() < deadline:
            if not engine.is_running():
                break
            time.sleep(1)
        if engine.is_running():
            _append_log(job_id, L("tune.log.server_relaunched"))
            continue
        break
    _wait_vram_freed(executor, target, job_id, used_before)


def _gpu_used_before(executor: Executor, target: Target) -> float:
    """停服前 GPU 已用显存 GB（释放判据）"""
    r = executor.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=8)
    vals = [float(x) for x in r.stdout.split() if x.strip().isdigit()]
    if vals:
        return round(sum(vals) / 1024, 1)
    if target.os == "linux":
        r = executor.run(
            "for d in /sys/class/drm/card*/device; do "
            "u=$(cat $d/mem_info_vram_used 2>/dev/null); "
            "[ -n \"$u\" ] && echo $u; done", timeout=8)
        best = 0.0
        for line in r.stdout.split():
            try:
                best = max(best, int(line) / 1024 ** 3)
            except ValueError:
                continue
        return round(best, 1)
    return 0.0


# ==================== 测速 ====================

def _wait_ready(executor: Executor, target: Target, timeout: int = 120) -> bool:
    """轮询目标机 /health 直到服务就绪"""
    deadline = time.time() + timeout
    cmd = (f'curl -s -o /dev/null -w "%{{http_code}}" --max-time 3 '
           f'http://127.0.0.1:{target.service_port}/health')
    while time.time() < deadline:
        r = executor.run(cmd, timeout=8)
        if r.stdout.strip() == "200":
            return True
        time.sleep(2)
    return False


def _curl_completion(executor: Executor, target: Target, payload: dict) -> Optional[dict]:
    """向目标机 llama-server 发一次 completion，返回解析后的 JSON 或 None。

    请求体统一走「write_file 落原始 UTF-8 JSON 到临时文件 + curl @file」：
    长 prompt 的 JSON 可达数十 KB，base64 内嵌进命令行会超 Windows cmd.exe
    8191 字符上限被截断（导致读到旧文件、测速失真），SFTP/本地写文件不受此限。
    """
    body = json.dumps(payload)
    url = f"http://127.0.0.1:{target.service_port}/completion"
    if target.os == "windows":
        json_path = "C:/temp/bench.json"
        # 先确保目录存在（短命令，不受长度限制）
        executor.run(
            'powershell -Command "New-Item -ItemType Directory -Force -Path C:\\temp | Out-Null"',
            timeout=15,
        )
        if not executor.write_file(body, json_path):
            return None
        cmd = (f'curl -s --max-time 120 -X POST {url} '
               f'-H "Content-Type: application/json" -d @{json_path}')
    else:
        json_path = "/tmp/bench.json"
        if not executor.write_file(body, json_path):
            return None
        cmd = (f"curl -s --max-time 120 -X POST {url} "
               f"-H 'Content-Type: application/json' -d @{json_path}")
    r = executor.run(cmd, timeout=130)
    if not r.stdout:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def _gpu_snapshot(executor: Executor, target: Target) -> dict:
    """测速瞬间抓一次 GPU 利用率/显存占用"""
    try:
        from .collectors import _collect_gpu
        return _collect_gpu(executor, target) or {}
    except Exception:
        return {}


def _cpu_mem_snapshot(executor: Executor, target: Target) -> dict:
    """测速瞬间抓一次 CPU 利用率/内存占用"""
    try:
        from .collectors import _collect_cpu_mem
        return _collect_cpu_mem(executor, target) or {}
    except Exception:
        return {}


def _bench_once(executor: Executor, target: Target, ctx_size: int) -> dict:
    """单次完整测速：短 prompt 测解码+TTFT，长 prompt 测预填充。
    返回 {decode, prefill, ttft_ms, gpu_util, gpu_mem_pct}。"""
    # 短 prompt：解码速度 + TTFT
    short = _curl_completion(executor, target, {
        "prompt": _BENCH_PROMPT,
        "n_predict": _BENCH_MAX_TOKENS,
        # 强制跑满 n_predict：不加这项时模型答完一句就 EOS，测量窗口塌缩到 ~24 token
        "ignore_eos": True,
        "temperature": 0,
        "stream": False,
    })
    decode = 0.0
    ttft_ms = 0.0
    if short:
        tm = short.get("timings", {})
        decode = float(tm.get("predicted_per_second", 0) or 0)
        # TTFT 近似 = 首 token 生成耗时：用 prompt 处理时间代表
        ttft_ms = float(tm.get("prompt_ms", 0) or 0)

    # 长 prompt：预填充速度
    # 加唯一随机前缀：llama 的 prefix cache 从序列头部匹配，前缀一变整段 cache 不命中，
    # 否则 warmup + 3 次重复发同一 prompt 时，第 2 次起 prompt_n 骤降、prefill 虚高成 ~40
    import uuid as _uuid
    long = _curl_completion(executor, target, {
        "prompt": f"[{_uuid.uuid4().hex[:16]}] " + _BENCH_LONG_PROMPT,
        "n_predict": 16,
        "temperature": 0,
        "stream": False,
    })
    prefill = 0.0
    if long:
        tm = long.get("timings", {})
        prefill = float(tm.get("prompt_per_second", 0) or 0)

    gpu = _gpu_snapshot(executor, target)
    cpu_mem = _cpu_mem_snapshot(executor, target)
    return {
        "decode": round(decode, 2),
        "prefill": round(prefill, 2),
        "ttft_ms": round(ttft_ms, 1),
        "gpu_util": gpu.get("utilization", 0),
        "gpu_mem_pct": gpu.get("memory_pct", 0),
        "cpu_pct": cpu_mem.get("cpu_pct", 0),
        "mem_used_gb": cpu_mem.get("memory_used_gb", 0),
        "mem_total_gb": cpu_mem.get("memory_total_gb", 0),
        "mem_pct": cpu_mem.get("memory_pct", 0),
    }


def _bench_median(executor: Executor, target: Target, ctx_size: int) -> dict:
    """预热 _BENCH_WARMUP 次（丢弃）+ 正式 _BENCH_REPEATS 次，各指标取中位数

    预热必须够：模型刚加载完的首个请求比稳态低约 18%（实测 56.12 → 68.67 t/s，
    首帧计算图构建等一次性开销），只预热 1 次会把这份冷启动损耗算进正式结果。
    """
    for _ in range(_BENCH_WARMUP):
        _bench_once(executor, target, ctx_size)  # 预热，丢弃
    runs = [_bench_once(executor, target, ctx_size) for _ in range(_BENCH_REPEATS)]
    return {
        "decode": round(median(r["decode"] for r in runs), 2),
        "prefill": round(median(r["prefill"] for r in runs), 2),
        "ttft_ms": round(median(r["ttft_ms"] for r in runs), 1),
        "gpu_util": round(median(r["gpu_util"] for r in runs), 1),
        "gpu_mem_pct": round(median(r["gpu_mem_pct"] for r in runs), 1),
        "cpu_pct": round(median(r["cpu_pct"] for r in runs), 1),
        "mem_used_gb": round(median(r["mem_used_gb"] for r in runs), 1),
        "mem_total_gb": round(median(r["mem_total_gb"] for r in runs), 1),
        "mem_pct": round(median(r["mem_pct"] for r in runs), 1),
    }


def _score(metrics: dict, goal: str) -> float:
    """按目标把多指标归一化加权成单一分数（越大越好）。
    用相对量纲：解码/预填充以各自最大值为参考，TTFT 取倒数。"""
    w = GOAL_WEIGHTS.get(goal, GOAL_WEIGHTS["latency"])
    decode = metrics.get("decode", 0)
    prefill = metrics.get("prefill", 0)
    ttft = metrics.get("ttft_ms", 0) or 1.0
    # decode<=0 说明这组根本没在生成（启动失败/立即 EOS/卡死），
    # 无论 prefill 报多高都不许进排名（Issue #1：CPU 组 decode 0.0 却拿了最高分）
    if decode <= 0:
        return 0.0
    # 归一化基准（经验上限，仅用于把不同量纲压到可比区间）
    score = (w["decode"] * decode +
             w["prefill"] * (prefill / 100.0) +   # 预填充常上千，缩 100 倍
             w["ttft"] * (1000.0 / ttft))         # TTFT 越小越好，取倒数
    return round(score, 3)


# ==================== 日志 / 任务 ====================

def _append_log(job_id: str, msg: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})


def get_job(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_active_jobs(target_id: str) -> list:
    """返回该目标机正在运行的调优任务摘要，供前端刷新后恢复轮询。"""
    with _LOCK:
        out = []
        for job in _JOBS.values():
            if job.get("target_id") != target_id:
                continue
            if job.get("status") != "running":
                continue
            logs = job.get("logs", [])
            out.append({
                "job_id": job["job_id"],
                "model": job.get("model", ""),
                "ctx_size": job.get("ctx_size", 0),
                "goal": job.get("goal", ""),
                "status": "running",
                "log_count": len(logs),
                "last_logs": logs[-8:],
                "result_count": len(job.get("results", [])),
            })
        return out


# ==================== 两阶段搜索 ====================

def _run_one(executor: Executor, target: Target, engine: LlamaCppAdapter,
             model_path: str, cfg: dict, ctx_size: int, job_id: str,
             tag: str) -> Optional[dict]:
    """启动一组配置→测速→停止，返回带 metrics 的结果；启动失败返回 None"""
    label = _cfg_label(cfg)
    _ensure_stopped(executor, target, engine, job_id)
    params = StartParams(model_path=model_path, extra_args=_args_list(cfg, target, ctx_size))
    ok, msg = engine.start(params)
    if not ok:
        _append_log(job_id, L("tune.log.run_fail", tag=tag, label=label, err=msg))
        return None
    if not _wait_ready(executor, target):
        _append_log(job_id, L("tune.log.run_timeout", tag=tag, label=label))
        engine.stop()
        return None
    metrics = _bench_median(executor, target, ctx_size)
    _ensure_stopped(executor, target, engine, job_id)
    _append_log(job_id, L("tune.log.run_result", tag=tag, label=label,
                          decode=metrics['decode'], prefill=metrics['prefill'],
                          gpu=metrics['gpu_util']))
    return {"config": cfg, "label": label, "metrics": metrics}


def _coarse_search(executor, target, engine, model_path, ctx_size,
                   vram: "_VRAMInfo", goal, job_id,
                   baseline_cfg: Optional[dict] = None,
                   baseline_gpu_ok: bool = False) -> Optional[dict]:
    """阶段一：搜离散主导因素 spec-type × cache-type。
    n-gpu-layers 默认 all（全进 GPU）；仅当 all 全部超显存时才降级用 0 兜底，
    避免把纯 CPU 跑大模型这种必然慢的组合当常规候选浪费测速时间。
    例外：基线刚在 GPU 上跑通过，估算公式再保守也不能推翻实测——仍测 GPU 组合，
    绝不静默降级 CPU（Issue #1：基线 82 t/s 跑在 GPU 上，coarse 却全跳 CPU）。
    """
    spec_options = list(SPEC_OPTIONS)
    # 基线用的复合投机模式（如 draft-mtp,ngram-mod,ngram-map-k4v）默认不在搜索
    # 空间里；模型带 MTP 头时把它加进来，否则调优结果永远匹配不上真实配置。
    base_spec = (baseline_cfg or {}).get("spec-type", "")
    if ("," in base_spec and base_spec not in spec_options
            and "mtp" in os.path.basename(model_path).lower()):
        spec_options.append(base_spec)

    def _build(ngl):
        out = []
        for spec in spec_options:
            for cache in CACHE_OPTIONS:
                cfg = _normalize_cfg(spec, cache, ngl,
                                     CONTINUOUS_GRID["batch-size"][1],
                                     CONTINUOUS_GRID["ubatch-size"][1], 3)
                if _fits_vram(cfg, vram, ctx_size):
                    out.append(cfg)
                else:
                    _append_log(job_id, L("tune.log.coarse_skip", cfg=_cfg_label(cfg)))
        return out

    candidates = _build("all")
    if not candidates and baseline_gpu_ok:
        _append_log(job_id, L("tune.log.vram_overridden"))
        candidates = [
            _normalize_cfg(spec, cache, "all",
                           CONTINUOUS_GRID["batch-size"][1],
                           CONTINUOUS_GRID["ubatch-size"][1], 3)
            for spec, cache in ((s, c) for s in spec_options for c in CACHE_OPTIONS)
        ]
    if not candidates:
        _append_log(job_id, L("tune.log.coarse_cpu"))
        candidates = _build("0")

    _append_log(job_id, L("tune.log.coarse_start", n=len(candidates)))
    scored = []
    for i, cfg in enumerate(candidates):
        r = _run_one(executor, target, engine, model_path, cfg, ctx_size,
                     job_id, f"coarse {i+1}/{len(candidates)}")
        if r:
            r["score"] = _score(r["metrics"], goal)
            scored.append(r)

    if not scored:
        return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    _append_log(job_id, L("tune.log.coarse_best", label=best['label'], score=best['score']))
    return best


def _fine_search(executor, target, engine, model_path, ctx_size,
                 vram: "_VRAMInfo", goal, job_id,
                 base_cfg: dict) -> dict:
    """阶段二：在 coarse 最优附近，对连续参数坐标下降收敛"""
    current = dict(base_cfg)
    cur = _run_one(executor, target, engine, model_path, current, ctx_size,
                   job_id, "fine base")
    if cur is None:
        return {"config": current, "label": _cfg_label(current),
                "metrics": {"decode": 0, "prefill": 0, "ttft_ms": 0,
                            "gpu_util": 0, "gpu_mem_pct": 0}, "score": 0}
    cur["score"] = _score(cur["metrics"], goal)
    best = cur

    spec_on = current.get("spec-type") != "off"
    tune_params = ["batch-size", "ubatch-size", "threads"]
    if spec_on:
        tune_params += ["spec-draft-n-max", "spec-draft-n-min", "spec-draft-p-min"]

    _append_log(job_id, L("tune.log.fine_start", params=tune_params))
    for param in tune_params:
        options = CONTINUOUS_GRID.get(param, [])
        improved = True
        while improved:
            improved = False
            # 按数值取值：spec-draft-p-min 是浮点，int() 会把 0.8 压成 0
            cur_val = _parse_num(best["config"].get(param, options[0]))
            idx = options.index(cur_val) if cur_val in options else 0
            # 向两侧各探一步
            for ni in (idx - 1, idx + 1):
                if ni < 0 or ni >= len(options):
                    continue
                trial = dict(best["config"])
                trial[param] = str(options[ni])
                if not _fits_vram(trial, vram, ctx_size):
                    continue
                r = _run_one(executor, target, engine, model_path, trial, ctx_size,
                             job_id, f"fine {param}={options[ni]}")
                if r is None:
                    continue
                r["score"] = _score(r["metrics"], goal)
                if r["score"] > best["score"]:
                    best = r
                    improved = True
                    _append_log(job_id, L("tune.log.fine_improved", param=param,
                                          value=options[ni], score=r['score']))
                    break
    _append_log(job_id, L("tune.log.fine_done", label=best['label'], score=best['score']))
    return best



# ==================== 读取用户「当前真实配置」 ====================
# 把正在运行的 llama-server 的命令行当作基线，而不是写死在前端的默认值。
# 只保留会影响速度的参数；模型路径、端口、采样参数、ctx、fit、flash-attn
# 由调优器自己控制或与速度无关，一律不带入。

_LIVE_KEYS = {
    "spec-type", "cache-type-k", "cache-type-v", "n-gpu-layers",
    "batch-size", "ubatch-size", "threads", "threads-batch", "parallel",
    "spec-draft-n-max", "spec-draft-n-min", "spec-draft-p-min",
    "spec-draft-ngl", "n-gpu-layers-draft", "gpu-layers-draft", "model-draft",
    "spec-ngram-mod-n-match", "spec-ngram-mod-n-min", "spec-ngram-mod-n-max",
    "split-mode", "tensor-split", "main-gpu", "numa", "rope-scaling",
    "keep", "override-kv",
    # 无值开关
    "mlock", "no-mmap", "no-warmup", "no-kv-offload", "kv-unified",
}
# 短参数名 / 别名 → 标准长名
_LIVE_ALIASES = {
    "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers",
    "ctk": "cache-type-k", "ctv": "cache-type-v",
    "b": "batch-size", "ub": "ubatch-size",
    "t": "threads", "tb": "threads-batch", "np": "parallel",
    "sm": "split-mode", "ts": "tensor-split", "mg": "main-gpu",
    "ngld": "n-gpu-layers-draft",
}
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def parse_llama_cmdline(tokens: List[str]) -> dict:
    """把 llama-server 的参数列表解析成 {标准长名: 值}（无值开关值为 ""）。

    返回 {"cfg": 影响速度的参数, "ctx": 命令行里的 ctx-size 或 None}。
    """
    cfg: dict = {}
    ctx = None
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        i += 1
        if not tok.startswith("-") or _NUM_RE.match(tok):
            continue  # 不是参数名（比如路径里含空格被拆开的残片）
        key = tok.lstrip("-")
        val = None
        if tok.startswith("--") and "=" in key:
            key, val = key.split("=", 1)
        key = _LIVE_ALIASES.get(key, key)
        if val is None and i < n:
            nxt = tokens[i]
            if (not nxt.startswith("-")) or _NUM_RE.match(nxt):
                val = nxt
                i += 1
        if key in ("ctx-size", "c") and val and val.isdigit():
            ctx = int(val)
            continue
        if key in _LIVE_KEYS:
            cfg[key] = val if val is not None else ""
    return {"cfg": cfg, "ctx": ctx}


def read_live_config(executor: Executor, target: Target, model: str = "") -> Optional[dict]:
    """读取目标机上正在运行的 llama-server 的真实参数。

    找不到（模型没加载 / Windows / 命令失败）返回 None，调用方回退到默认基线。
    多个 llama-server 时优先取命令行里含所选模型文件名的那个。
    """
    if target.os == "windows":
        return None
    r = executor.run("ps -eo args=", timeout=10)
    if not r.ok or not r.stdout:
        return None
    found = []
    for line in r.stdout.splitlines():
        line = line.strip()
        try:
            toks = shlex.split(line)
        except ValueError:
            continue
        if toks and os.path.basename(toks[0]) == "llama-server":
            found.append((line, toks))
    if not found:
        return None
    pick = next((f for f in found if model and model in f[0]), found[0])
    parsed = parse_llama_cmdline(pick[1][1:])
    if not parsed["cfg"]:
        return None
    parsed["raw"] = pick[0]
    return parsed


# ==================== 主流程 ====================

def start_tune(target_id: str, model: str, ctx_size: int = 8192,
               goal: str = "latency", baseline_cfg: Optional[dict] = None,
               model_size_gb: float = 0.0) -> dict:
    """启动两阶段调优任务。
    baseline_cfg：用户原始参数（dict），作为基线先测一组对比。
    model_size_gb：模型大小，用于显存预检；缺省按 0 跳过预检。
    """
    # 后台线程不继承请求上下文，先捕获发起时的界面语言，进 worker 再设回去
    lang = get_lang()
    target = get_target(target_id)
    if not target:
        return {"ok": False, "message": L("tune.err.no_target")}
    if not target.engine_path:
        return {"ok": False, "message": L("tune.err.no_engine")}
    if getattr(target, "engine_type", "llama_cpp") != "llama_cpp":
        return {"ok": False, "message": L("tune.err.llama_only")}
    if not target.models_dir or not model:
        return {"ok": False, "message": L("tune.err.no_model")}
    if ctx_size < 1024:
        return {"ok": False, "message": L("tune.err.ctx_small")}

    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id, "target_id": target_id, "model": model,
            "ctx_size": ctx_size, "goal": goal,
            "status": "running", "logs": [], "results": [],
            "baseline": None, "best": None, "error": "",
        }

    def _worker():
        nonlocal model_size_gb
        executor = None
        set_lang(lang)
        try:
            from .executor import make_executor
            from .collectors import path_join
            executor = make_executor(target)
            engine = LlamaCppAdapter(executor, target)

            if not engine.check_installed():
                _fail(job_id, L("tune.fail.engine_missing"))
                _append_log(job_id, L("tune.log.no_engine"))
                return

            model_path = path_join(target, target.models_dir, model)

            # 取目标机显存；模型大小未传则自动探测（用于显存预检）
            gpu_vram_gb = _get_gpu_vram(executor, target)
            if model_size_gb <= 0:
                model_size_gb = _get_model_size_gb(executor, target, model_path)
            # 从 GGUF 文件头读模型真实结构（层数/注意力布局），让 KV 估算
            # 按模型实际大小算，而不是按"27B 标准件"猜
            meta = _read_gguf_meta(executor, target, model_path)
            kv_layers = _kv_layers_from_meta(meta)
            kv_heads = int(meta.get("kv_heads") or 0)
            head_count = int(meta.get("head_count") or 0)
            hidden = int(meta.get("hidden") or 0)
            kv_head_dim = (hidden // head_count) if (hidden and head_count) else 0
            vram = _VRAMInfo(
                model_size_gb=model_size_gb,
                gpu_vram_gb=gpu_vram_gb,
                kv_layers=kv_layers,
                kv_heads=kv_heads,
                kv_head_dim=kv_head_dim,
                bytes_per_kv_token=kv_heads * kv_head_dim,
            )
            _append_log(job_id, L("tune.log.hw", vram=f"{gpu_vram_gb:.1f}", model=model,
                                  size=f"{model_size_gb:.1f}", ctx=ctx_size,
                                  goal=goal_label(goal)))
            all_results = []
            baseline_gpu_ok = False

            # 基线：优先用目标机上正在运行的 llama-server 的真实参数；
            # 读不到（模型没加载等）才回退到前端传来的默认基线。
            # 必须在第一次 engine.stop() 之前读取，否则运行中的服务已被停掉。
            base_cfg = baseline_cfg
            if baseline_cfg:
                live = read_live_config(executor, target, model)
                if live:
                    base_cfg = live["cfg"]
                    _append_log(job_id, "Captured your running llama-server config (used as baseline):")
                    _append_log(job_id, "  raw: " + live["raw"])
                    _append_log(job_id, "  baseline: " + _cfg_label(base_cfg))
                    if live["ctx"] and live["ctx"] != ctx_size:
                        _append_log(job_id, f"  note: your server runs ctx-size {live['ctx']} "
                                            f"but this tuning is fixed at {ctx_size}")
                else:
                    _append_log(job_id, "No running llama-server found; using the built-in default "
                                        "baseline. Load your model first (send one request) to use "
                                        "your real config.")
            if base_cfg:
                _append_log(job_id, L("tune.log.baseline_test"))
                b = _run_one(executor, target, engine, model_path, base_cfg,
                             ctx_size, job_id, "baseline")
                if b:
                    b["score"] = _score(b["metrics"], goal)
                    all_results.append(b)
                    with _LOCK:
                        _JOBS[job_id]["baseline"] = b
                    _append_log(job_id, L("tune.log.baseline_score", score=b['score']))
                    # 基线真的在 GPU 上跑起来了（利用率可观）：后续任何"估算说
                    # 放不下"都不许推翻这个实测结论——不允许静默降级成纯 CPU 测试
                    baseline_gpu_ok = b["metrics"].get("gpu_util", 0) >= 5
                    # GGUF 头读不到结构（自定义量化的非标准布局）时，用基线
                    # 真实占用的显存校准估算器，让后续预检以实测为准
                    if baseline_gpu_ok and not vram.meta_ok:
                        if _calibrate_vram(vram, b["metrics"], base_cfg, ctx_size):
                            _append_log(job_id, L("tune.log.vram_calibrated",
                                                  kv=f"{vram.bytes_per_kv_token}"))

            # 阶段一 coarse
            coarse_best = _coarse_search(executor, target, engine, model_path,
                                         ctx_size, vram, goal, job_id,
                                         baseline_cfg=base_cfg,
                                         baseline_gpu_ok=baseline_gpu_ok)
            if not coarse_best:
                _fail(job_id, L("tune.fail.no_config"))
                with _LOCK:
                    _JOBS[job_id]["results"] = all_results
                return
            all_results.append(coarse_best)

            # 阶段二 fine
            fine_best = _fine_search(executor, target, engine, model_path,
                                     ctx_size, vram, goal, job_id,
                                     coarse_best["config"])
            all_results.append(fine_best)

            # 最终推荐 = fine 收敛结果；与基线对比
            final_best = fine_best if fine_best["score"] > 0 else coarse_best
            _finalize(job_id, all_results, final_best)
        except Exception as e:
            _fail(job_id, str(e))
            _append_log(job_id, L("tune.log.exception", err=e))
        finally:
            try:
                if executor:
                    LlamaCppAdapter(executor, target).stop()
            except Exception:
                pass
            if executor:
                executor.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True, "job_id": job_id}


def _get_model_size_gb(executor: Executor, target: Target, model_path: str) -> float:
    """探测目标机上模型文件实际大小（GB），用于显存预检"""
    if target.os == "windows":
        cmd = (f'powershell -Command "if(Test-Path \'{model_path}\')'
               f'{(chr(123))}(Get-Item \'{model_path}\').Length{(chr(125))}else{{0}}"')
    else:
        cmd = f'stat -Lc %s "{model_path}" 2>/dev/null || stat -c %s "{model_path}" 2>/dev/null || echo 0'
    r = executor.run(cmd, timeout=10)
    digits = "".join(c for c in r.stdout if c.isdigit())
    if digits:
        return round(int(digits) / (1024 ** 3), 1)
    return 0.0


def _get_gpu_vram(executor: Executor, target: Target) -> float:
    try:
        from .collectors import _detect_gpu_static, _detect_memory_static
        gpu = _detect_gpu_static(executor, target)
        if gpu:
            v = gpu.get("total_memory_gb", 0)
            if v > 0:
                return float(v)
            if gpu.get("unified"):
                mem = _detect_memory_static(executor, target)
                return mem.get("total_gb", 0) * 0.75
    except Exception:
        pass
    return 0.0


def _fail(job_id: str, err: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = err


def _finalize(job_id: str, results: List[dict], best: dict):
    # 标注推荐
    for r in results:
        r["recommended"] = (r["label"] == best["label"])
    with _LOCK:
        job = _JOBS[job_id]
        job["status"] = "success"
        job["results"] = results
        job["best"] = best
        _tid, _model, _ctx = job["target_id"], job["model"], job["ctx_size"]
    _append_log(job_id, L("tune.log.done", label=best['label'], score=best['score']))
    # 落盘最近调优参数，供部署页作为默认参数回填
    try:
        from .tune_history import save_latest
        save_latest(_tid, _model, _ctx, best.get("config", {}),
                    source="tuner", score=best.get("score", 0))
    except Exception as e:
        _append_log(job_id, L("tune.log.save_fail", err=e))
