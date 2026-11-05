"""智能调优压测服务（两阶段搜索版）

在用户当前硬件配置下，为目标机上的某个模型搜索最优推理参数。

设计要点：
  1. 参数分三类：
     - 约束类（用户/硬件定死）：模型、端口、ctx 下限 → 不搜，作硬约束
     - 离散高影响类：spec-type / cache-type-k,v / n-gpu-layers → coarse 阶段搜
     - 连续微调类：batch-size / ubatch-size / spec-draft-n-max → fine 阶段坐标下降
  2. 显存可行性预检：**以实测为准**。先跑基线拿到"服务驻留时的已用显存"，减去
     空载占用和权重实占体积，反推每 token 的 KV 字节数，再用它判断别的上下文/量化
     组合放不放得下。没校准过就一律放行，交给真实启动判定（详见 _Vram）
  3. 两阶段搜索：coarse 定主导因素 → fine 在最优组合附近收敛
  4. 可信测速：预热 2 次 + 正式 3 次取中位数（解码测速用 ignore_eos 跑满固定窗口，
     避免模型提前 EOS 导致测量窗口塌缩），记录解码/预填充/TTFT/GPU 利用率
  5. 基线对比：以用户原始参数为 baseline 先测，输出"推荐 vs 当前"

优化目标可选：latency(端到端体感,默认) / throughput(纯解码吞吐) / prefill(长文本预填充)。
无引擎/无模型时明确报错，绝不返回模拟数据。
"""

import threading
import time
import json
import uuid
from dataclasses import dataclass
from statistics import median
from typing import Optional, List, Dict, Tuple

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
# 连续多少次「一个 token 都没产出、预填充也没有」就判定服务已崩、立刻放弃本轮 bench。
# 服务崩掉时若不快速失败，预热 2 次 + 正式 3 次、每次两个请求各等满 --max-time 120，
# 最坏静默挂死 20 分钟而前端只看到 running（真机 job 9fa06f07 实测）。
_BENCH_DEAD_STREAK = 2
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
        elif k == "spec-type":
            # 内部用 "off" 表示「不启用投机解码」，但 llama.cpp 的 --spec-type 只认
            # none / draft-* / ngram-*，根本没有 "off"。
            # 直接拼成 --spec-type off 会让 llama-server 在参数解析阶段就退出（实测报文：
            # 'unknown speculative type: off'），_wait_ready 白等满超时，
            # 最终被误报成「启动超时(可能显存不足)」——把参数错误归因成了显存不足。
            args += ["--spec-type", "none" if str(v) == "off" else str(v)]
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


# ============ 读取用户真实在跑的配置（issue #6 问题 6） ============

# 只认「会改变推理速度」的开关。模型路径 / ctx / 端口 / host / metrics 由 tuner
# 自己接管（必须固定，否则各组之间不可比）；--fit / --flash-attn 属于显存与上下文
# 策略（issue #6 问题 8），基线里也刻意不覆盖：所有候选必须跑在同一策略下才公平。
_LIVE_CFG_KEYS = {
    "n-gpu-layers", "cache-type-k", "cache-type-v", "batch-size", "ubatch-size",
    "spec-type", "spec-draft-n-max", "spec-draft-n-min", "spec-draft-p-min",
    "threads", "threads-batch", "tensor-split", "split-mode", "main-gpu",
    "no-kv-offload", "mlock", "no-mmap",
}

# llama-server 的短选项与别名，统一映射回 _args_list 里的长键名
_LIVE_CFG_ALIAS = {
    "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers",
    "n_gpu_layers": "n-gpu-layers",
    "ctk": "cache-type-k", "ctv": "cache-type-v",
    "b": "batch-size", "ub": "ubatch-size",
    "t": "threads", "tb": "threads-batch",
    "sm": "split-mode", "mg": "main-gpu",
}


def _parse_llama_cmdline(cmdline: str) -> Dict[str, str]:
    """把 llama-server 命令行还原成 _args_list 能吃的 cfg dict。

    只处理 `--key value` / `-key value`。裸开关（如 --no-mmap、--mlock，后面直接跟
    另一个选项或结束）保留为 "on"，因为它们本身是显存/内存策略的一部分。
    """
    import shlex
    try:
        toks = shlex.split(cmdline or "")
    except ValueError:
        return {}
    out: Dict[str, str] = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not (tok.startswith("-") and len(tok) > 1):
            i += 1
            continue
        key = tok.lstrip("-")
        key = _LIVE_CFG_ALIAS.get(key, key)
        if key not in _LIVE_CFG_KEYS:
            i += 1
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if nxt and not nxt.startswith("-"):
            out[key] = nxt
            i += 2
        else:
            out[key] = "on"
            i += 1
    return out


def _read_live_cfg(executor: Executor, target: Target) -> Optional[dict]:
    """读目标机上正在运行的 llama-server 命令行，解析成基线配置。

    这是 issue #6 问题 6：基线应当来自用户真实在跑的进程，而不是前端硬编码的
    DEFAULT_BASELINE。读不到（没在跑 / 命令失败 / 解析为空）返回 None，
    由调用方回落到内置基线。
    """
    if target.os == "windows":
        from .collectors import _ps
        cmd = _ps("$p=Get-CimInstance Win32_Process | "
                  "Where-Object {$_.Name -eq 'llama-server.exe'} | "
                  "Select-Object -First 1 -ExpandProperty CommandLine")
        timeout = 25
    else:
        cmd = "ps -eo args | grep -m1 '[l]lama-server'"
        timeout = 10
    r = executor.run(cmd, timeout=timeout)
    if not r.ok:
        return None
    return _parse_llama_cmdline(r.stdout) or None


# ==================== 显存可行性预检 ====================

# KV cache 量化每元素字节数（同量化下 KV 占用与字节数成正比，用于跨量化换算）
_CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

# 校准探针的第二档上下文倍数：两档差值越大，MB 级读数噪声相对影响越小。
# ctx 8192 时差值为 3 份 KV（27B q4_0 约 1.5 GB），足够压过噪声。
CALIB_CTX_MULTIPLIER = 4
# 两档显存读数差值低于此值时视为噪声主导，宁可放弃校准（预检放行）也不算斜率
CALIB_MIN_KV_DELTA_GB = 0.15


@dataclass
class _Vram:
    """显存预检的全部依据。

    设计原则：**只用实测，不猜模型结构。**
    旧实现用硬编码的 64 层 × KV 维度 8192 套公式，对混合注意力模型（Qwen 系列等
    滑窗层不存逐 token KV）会高估 3~4 倍，在 131072 上下文下把能跑的组合全判成
    "显存不足"。改成从 GGUF 头解析层数/头数也不行：GGUF 的 value type 编号、
    字符串编码（uint64 长度前缀）都有自己的规范，手写解析器一旦错位就静默失效，
    且非标准导出的自定义量化文件里这些字段未必可信。

    因此这里只认两样东西：
    1) model_size_gb：权重文件实际大小（stat 测得）；
    2) kv_bytes_per_token：基线在 GPU 上真跑通过之后，用
       "实测已用显存 − 权重" 反推出的每 token KV 字节数（f16 基准）。

    kv_bytes_per_token <= 0 表示还没校准过，此时预检一律放行（advisory），
    让真实启动去判定——错误的猜测式拦截比放行更糟。
    """
    model_size_gb: float = 0.0
    gpu_vram_gb: float = 0.0
    kv_bytes_per_token: float = 0.0   # f16 基准；0 = 未校准
    calibrated_ctx: int = 0           # 校准时的上下文长度
    idle_used_gb: float = 0.0         # 空载时 GPU 已用显存（区分"别人的占用"）
    cache_bytes_at_calib: float = 2.0
    # 试启动预检开关。默认关 = 校准值只做参考、不用来拦截（两轮真机证明它测不准：
    # 同配置两次差 17%，实测增量只有理论 1/3）。开启后改为真启动一次问结果。
    precheck_enabled: bool = False

    @property
    def calibrated(self) -> bool:
        return self.kv_bytes_per_token > 0 and self.calibrated_ctx > 0

    def estimate_gb(self, ctx_size: int, cache_type: str) -> float:
        """按校准结果估算重量 + KV 占用；未校准时返回 0（调用方不得据此拦截）"""
        if not self.calibrated:
            return 0.0
        cache_b = _CACHE_BYTES.get(cache_type, 2.0)
        scale = cache_b / self.cache_bytes_at_calib
        kv_gb = (self.kv_bytes_per_token * scale * ctx_size) / (1024 ** 3)
        return self.model_size_gb + kv_gb


def _gpu_mem_used_gb(executor: Executor, target: Target) -> float:
    """目标机 GPU 已用显存 GB；读不到返回 0.0

    多卡统一取 **最大值**（不是求和）：tuner 的所有候选都只往单卡装，
    用累加值会把两张卡的空闲显存当成一块连续空间，判据整体失真。
    """
    r = executor.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=8)
    vals = [float(x) for x in r.stdout.split() if x.strip().isdigit()]
    if vals:
        return round(max(vals) / 1024, 2)
    if target.os != "windows":
        r = executor.run(
            "for d in /sys/class/drm/card*/device; do "
            "u=$(cat $d/mem_info_vram_used 2>/dev/null); "
            "[ -n \"$u\" ] && echo $u; done", timeout=8)
        best = 0.0
        for line in r.stdout.split():
            try:
                best = max(best, int(line))
            except ValueError:
                continue
        if best:
            return round(best / 1024 ** 3, 2)
    return 0.0


def _calibrate(vram: _Vram, probes: List[Tuple[int, float]],
               cache_type: str) -> bool:
    """用同一配置、两个不同 ctx 的实测显存求差，反推纯 KV 斜率。

    为什么不能用单点（旧做法）：llama-server 起来的显存里，除了 KV 还有一批
    **不随 ctx 增长**的固定开销——MTP draft、compute buffer、CUDA context。
    旧实现把 used − idle − model 全当作 KV，等于把这些固定开销除以 ctx 变成
    "每 token 字节数"，斜率被放大数倍，ctx 越大错得越离谱。

    真机实测（27B Q4_K_M / 4090 / ctx 8192 / q4_0）：
        无 MTP 探针  17902 − 429 − 17043 =  430 MB  ≈ 理论 KV 512 MB（吻合）
        带 MTP 基线  20316 − 429 − 17043 = 2844 MB  → 多算 2414 MB
    单点反推因此报出 1431306 B/token，真值约 262144，虚高 5.46 倍，
    4090 上本该跑得下的 f16 组合被误判显存不足而跳过。

    两点相减时，两档都存在的固定开销自动消掉，只剩随 ctx 变化的部分：
        (used_b − used_a) × 1024³ / (ctx_b − ctx_a) = 纯 KV 字节/token
    因此这里**不再需要 idle_used 和 model_size**，它们在相减中同样抵消。
    副产品：校准不再依赖两个容易读偏的值（模型软链、别的进程占卡）。

    probes: [(ctx, 已用显存 GB), ...]，至少两个 ctx 不同的点。
    """
    pairs = [(int(c), float(u)) for c, u in probes if c > 0 and u > 0]
    if len(pairs) < 2:
        return False
    pairs.sort()
    (ctx_a, used_a), (ctx_b, used_b) = pairs[0], pairs[-1]
    d_ctx = ctx_b - ctx_a
    d_used = used_b - used_a
    if d_ctx <= 0:
        return False
    # 差值太小：MB 级读数噪声会主导，算出的斜率不可信，宁可不校准（预检放行）
    if d_used < CALIB_MIN_KV_DELTA_GB:
        return False
    kv_bytes_actual = d_used * (1024 ** 3) / d_ctx
    cache_b = _CACHE_BYTES.get(cache_type, 2.0)
    vram.kv_bytes_per_token = kv_bytes_actual * (2.0 / cache_b)
    vram.calibrated_ctx = ctx_b
    vram.cache_bytes_at_calib = 2.0
    return True


def _probe_used_gb(executor: Executor, target: Target, engine: LlamaCppAdapter,
                   model_path: str, cfg: dict, ctx_size: int, job_id: str,
                   tag: str, vram: Optional[_Vram] = None) -> float:
    """启动一次、只读模型驻留后的显存，然后停服；**不跑测速**。

    校准只需要显存读数，不需要速度。复用 _run_one 会白跑 3 遍 benchmark，
    而加载权重才是耗时主体——省掉 bench 让第二档探针代价小得多。
    """
    engine.stop()
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    params = StartParams(model_path=model_path,
                         extra_args=_args_list(cfg, target, ctx_size))
    ok, msg = engine.start(params)
    if not ok:
        _append_log(job_id, L("tune.log.calib_probe_fail", tag=tag, err=msg))
        return 0.0
    if not _wait_ready(executor, target, engine=engine):
        _append_log(job_id, L("tune.log.calib_probe_fail", tag=tag,
                              err="服务进程已退出或就绪超时"))
        engine.stop()
        return 0.0
    used_gb = _gpu_mem_used_gb(executor, target)
    engine.stop()
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    _append_log(job_id, L("tune.log.calib_probe", tag=tag, ctx=ctx_size,
                          used=f"{used_gb:.2f}"))
    return used_gb


def _precheck_start(executor: Executor, target: Target, engine: LlamaCppAdapter,
                    model_path: str, cfg: dict, ctx_size: int, job_id: str,
                    vram: Optional[_Vram] = None) -> bool:
    """试启动预检：不猜斜率，直接把这组参数真正拉起一次。

    为什么要这个：两点差分实测只能测到理论 KV 增量的 1/3（信号 0.5 GB，
    读数噪声就有 0.09 GB），算出来的斜率两次差 17%，不可复现，不能拿来拦截。
    与其猜，不如直接问一句「这组起得来吗」——启动成功就是最硬的证据。

    代价：每组多一次真实加载（27B 约 30~60 秒），所以默认关闭，
    只有显式开启 vram.precheck_enabled 时才走这条路。
    """
    engine.stop()
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    params = StartParams(model_path=model_path,
                         extra_args=_args_list(cfg, target, ctx_size))
    ok, msg = engine.start(params)
    if not ok:
        _append_log(job_id, L("tune.log.precheck_fail", err=msg))
        return False
    ready = _wait_ready(executor, target, engine=engine)
    engine.stop()
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    if not ready:
        _append_log(job_id, L("tune.log.precheck_fail",
                              err="服务未就绪或进程已退出"))
    return ready


def _fits_vram(cfg: dict, vram: _Vram, ctx_size: int) -> bool:
    """估算这组参数放不放得下——**仅供参考，默认不再用于拦截**。

    历史教训：这里曾经拿校准值做硬拦截，结果两层错误叠加，把能跑的配置全拦掉。
    先是硬编码模型结构公式对混合注意力模型高估 3~4 倍；改用两点差分后，
    实测斜率又只有真值的 1/3（信号 0.5 GB vs 理论 1.61 GB），偏到了另一头。

    两轮真机数据（job f6c25914 / d50f99b8）证明这个量测不准：同配置两次算出
    90876 和 106605，差 17%，而读数噪声就占了信号的 15%。

    因此语义改为：默认一律放行，让真实启动去判定。只有显式开启
    vram.precheck_enabled（走 _precheck_start 真启动）时才做筛选。
    """
    if not vram.precheck_enabled:
        return True
    if cfg.get("n-gpu-layers") == "0":
        return True
    if not vram.calibrated or vram.gpu_vram_gb <= 0:
        return True
    est = vram.estimate_gb(ctx_size, cfg.get("cache-type-k", "f16"))
    if est <= 0:
        return True
    # 留 6% 余量给激活值/显存碎片（比原来的 10% 宽松：校准值本身已含实测开销）
    return est <= vram.gpu_vram_gb * 0.94


def _wait_vram_freed(executor: Executor, target: Target, job_id: str,
                     idle_used_gb: float, deadline_s: int = 90) -> float:
    """停服后等到显存回落到接近空载水平，再放行下一组测试；返回当前已用 GB。

    旧实现是固定 `time.sleep(2)`：27B 的显存归还往往要十几秒，抢跑会在
    "显存已满"的卡上启动下一组，表现成一连串莫名的启动超时。

    判据用 idle_used_gb（本次调优开始前的空载占用）而不是 0：桌面环境、
    别的常驻服务都算在里面，拿 0 当目标会永远等不满、每组都白等到超时。
    读不到显存（无 GPU / 命令失败）时直接返回，不做无意义的等待。
    """
    if idle_used_gb <= 0:
        return 0.0
    limit = idle_used_gb + 1.0          # 1 GB 容差：驱动自身占用会小幅抖动
    deadline = time.time() + deadline_s
    warned = False
    stall = 0
    used = _gpu_mem_used_gb(executor, target)
    last = used
    while time.time() < deadline:
        if used <= 0:
            return 0.0                  # 读不到显存，别等了
        if used <= limit:
            return used
        # 连续 4 次（约 12 秒）毫无回落 → 是别的进程稳定占着，不是我们的残留：
        # 继续等只会让每组白等到 90 秒超时，按时长把整轮调优拖垮。
        stall = stall + 1 if used >= last - 0.2 else 0
        last = used
        if stall >= 4:
            _append_log(job_id, L("tune.log.vram_still_busy"))
            return used
        if not warned:
            _append_log(job_id, L("tune.log.vram_wait"))
            warned = True
        time.sleep(3)
        used = _gpu_mem_used_gb(executor, target)
    if warned:
        _append_log(job_id, L("tune.log.vram_still_busy"))
    return used


# ==================== 测速 ====================

def _wait_ready(executor: Executor, target: Target, timeout: int = 120,
                engine=None) -> bool:
    """轮询目标机 /health 直到服务就绪。

    传入 engine 时会额外核对「自己启动的那个进程是否还活着」。
    原因：参数非法（例如把内部标记 off 原样拼成 --spec-type off）会让 llama-server
    在参数解析阶段立刻退出，端口永远不会通；只盯 /health 就会干等满 timeout，
    把「参数错了」误报成「启动超时(显存不足)」——真机 job f6c25914 就因此把
    coarse 六组里的三组判成显存不足，而当时显存只用了 684 MiB。
    """
    deadline = time.time() + timeout
    cmd = (f'curl -s -o /dev/null -w "%{{http_code}}" --max-time 3 '
           f'http://127.0.0.1:{target.service_port}/health')
    checks = 0
    while time.time() < deadline:
        r = executor.run(cmd, timeout=8)
        if r.stdout.strip() == "200":
            return True
        checks += 1
        # 每 3 次探测核对一次进程存活：_list_pids 要走一次远程查询，不宜过密。
        # 留出几次冗余是必要的——刚启动时进程可能还没被枚举到。
        if engine is not None and checks % 3 == 0:
            pid = getattr(engine, "_pid", None)
            if pid:
                try:
                    alive = pid in engine._list_pids()
                except Exception:
                    alive = True  # 查不到不能当死亡，否则会把慢启动误判成失败
                if not alive:
                    return False
        time.sleep(2)
    return False


def _health_ok(executor: Executor, target: Target, timeout: int = 3) -> bool:
    """目标机 /health 是否可用。发重请求前用它快速判断服务死活。

    没有这道探测时，服务一旦崩掉（进程还在、端口已不听），每个 completion
    请求都要等满 --max-time 120：预热 2 次 + 正式 3 次，每次两个请求，
    最坏静默挂死 20 分钟，前端只看到 status=running（真机 job 9fa06f07 实测）。
    """
    cmd = (f'curl -s -o /dev/null -w "%{{http_code}}" --max-time {timeout} '
           f'http://127.0.0.1:{target.service_port}/health')
    r = executor.run(cmd, timeout=timeout + 5)
    return r.stdout.strip() == "200"


def _curl_completion(executor: Executor, target: Target, payload: dict) -> Optional[dict]:
    """向目标机 llama-server 发一次 completion，返回解析后的 JSON 或 None。

    请求体统一走「write_file 落原始 UTF-8 JSON 到临时文件 + curl @file」：
    长 prompt 的 JSON 可达数十 KB，base64 内嵌进命令行会超 Windows cmd.exe
    8191 字符上限被截断（导致读到旧文件、测速失真），SFTP/本地写文件不受此限。
    """
    # 服务已经崩掉就直接返回，不要让 --max-time 120 反复空等
    if not _health_ok(executor, target):
        return None
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
    # 空结果：decode=0 时 _score 给 0 分，与「这组跑不动」的既有语义一致
    empty = {"decode": 0.0, "prefill": 0.0, "ttft_ms": 0.0, "gpu_util": 0.0,
             "gpu_mem_pct": 0.0, "cpu_pct": 0.0, "mem_used_gb": 0.0,
             "mem_total_gb": 0.0, "mem_pct": 0.0}

    def _dead(m: dict) -> bool:
        """一个 token 都没生成、预填充也没有 → 不是慢，是服务根本没响应"""
        return m["decode"] <= 0 and m["prefill"] <= 0

    dead_streak = 0
    for _ in range(_BENCH_WARMUP):
        m = _bench_once(executor, target, ctx_size)  # 预热，丢弃
        dead_streak = dead_streak + 1 if _dead(m) else 0
        if dead_streak >= _BENCH_DEAD_STREAK:
            return empty

    runs = []
    for _ in range(_BENCH_REPEATS):
        m = _bench_once(executor, target, ctx_size)
        dead_streak = dead_streak + 1 if _dead(m) else 0
        if dead_streak >= _BENCH_DEAD_STREAK:
            return empty
        runs.append(m)
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
    # decode<=0 说明这组压根没在生成（启动失败 / 立即 EOS / 卡死）。
    # 此时 prefill 仍可能报出很高值（纯 CPU 组出现过 decode 0.0、prefill 255），
    # 不加守卫的话，一个跑不出 token 的配置能靠 prefill 拿到最高分并被推荐。
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
             tag: str, vram: Optional[_Vram] = None) -> Optional[dict]:
    """启动一组配置→测速→停止，返回带 metrics 的结果；启动失败返回 None"""
    label = _cfg_label(cfg)
    engine.stop()
    # 停服后等显存真的归还再启动下一组：27B 释放显存常要几秒到十几秒，
    # 固定 sleep 2 秒会让下一组在"卡还没清空"时启动，表现成一连串莫名超时。
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    params = StartParams(model_path=model_path, extra_args=_args_list(cfg, target, ctx_size))
    ok, msg = engine.start(params)
    if not ok:
        _append_log(job_id, L("tune.log.run_fail", tag=tag, label=label, err=msg))
        return None
    if not _wait_ready(executor, target, engine=engine):
        _append_log(job_id, L("tune.log.run_fail", tag=tag, label=label,
                              err="服务进程已退出（多为参数非法，详见服务端日志）"))
        engine.stop()
        return None
    # 服务已就绪、模型已驻留显存，此刻读到的已用显存才是校准可用的样本
    used_gb = _gpu_mem_used_gb(executor, target)
    metrics = _bench_median(executor, target, ctx_size)
    engine.stop()
    if vram is not None:
        _wait_vram_freed(executor, target, job_id, vram.idle_used_gb)
    else:
        time.sleep(2)
    _append_log(job_id, L("tune.log.run_result", tag=tag, label=label,
                          decode=metrics['decode'], prefill=metrics['prefill'],
                          gpu=metrics['gpu_util']))
    return {"config": cfg, "label": label, "metrics": metrics,
            "gpu_mem_used_gb": used_gb}


def _coarse_search(executor, target, engine, model_path, ctx_size,
                   vram: _Vram, goal, job_id,
                   baseline_gpu_ok: bool = False) -> Optional[dict]:
    """阶段一：搜离散主导因素 spec-type × cache-type。

    n-gpu-layers 默认 all（全进 GPU），且**不再自动降级到纯 CPU**：
    旧版一旦估算说"放不下"就整批换成 n-gpu-layers=0 继续测，纯 CPU 组跑
    27B 每组要十几分钟，还常常 decode 0.0，却因为 prefill 很高而被判为最优
    —— 用户看到的就是"调优结果建议用 CPU 跑大模型"（issue #5）。
    现在只在"估算放不下、但基线刚在 GPU 上跑通"时按原计划测 GPU（估算不可信），
    其余情况失败就如实报错，不要用 CPU 结果凑数。
    """
    def _build(ngl):
        ok, skipped = [], []
        for spec in SPEC_OPTIONS:
            for cache in CACHE_OPTIONS:
                cfg = _normalize_cfg(spec, cache, ngl,
                                     CONTINUOUS_GRID["batch-size"][1],
                                     CONTINUOUS_GRID["ubatch-size"][1], 3)
                if vram.precheck_enabled:
                    # 预检模式：不算斜率，直接真启动一次，起得来就参赛
                    if _precheck_start(executor, target, engine, model_path,
                                       cfg, ctx_size, job_id, vram):
                        ok.append(cfg)
                    else:
                        skipped.append(cfg)
                        _append_log(job_id, L("tune.log.precheck_skip",
                                              cfg=_cfg_label(cfg)))
                elif _fits_vram(cfg, vram, ctx_size):
                    ok.append(cfg)
                else:
                    skipped.append(cfg)
                    _append_log(job_id, L("tune.log.coarse_skip", cfg=_cfg_label(cfg)))
        return ok, skipped

    candidates, skipped = _build("all")
    if not candidates and skipped and baseline_gpu_ok:
        # 基线用真实启动证明了这些配置放得下，估算器没有资格拦它们
        _append_log(job_id, L("tune.log.vram_overridden"))
        candidates = skipped
    if not candidates:
        _append_log(job_id, L("tune.log.coarse_no_gpu"))
        return None

    _append_log(job_id, L("tune.log.coarse_start", n=len(candidates)))
    scored = []
    for i, cfg in enumerate(candidates):
        r = _run_one(executor, target, engine, model_path, cfg, ctx_size,
                     job_id, f"coarse {i+1}/{len(candidates)}", vram)
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
                 vram: _Vram, goal, job_id,
                 base_cfg: dict) -> dict:
    """阶段二：在 coarse 最优附近，对连续参数坐标下降收敛"""
    current = dict(base_cfg)
    cur = _run_one(executor, target, engine, model_path, current, ctx_size,
                   job_id, "fine base", vram)
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
                             job_id, f"fine {param}={options[ni]}", vram)
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


# ==================== 主流程 ====================

def start_tune(target_id: str, model: str, ctx_size: int = 8192,
               goal: str = "latency", baseline_cfg: Optional[dict] = None,
               model_size_gb: float = 0.0, vram_precheck: bool = False) -> dict:
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
            # 空载已用显存：既是"等显存释放"的目标水位，也是校准时该扣掉的底噪
            idle_used_gb = _gpu_mem_used_gb(executor, target)
            vram = _Vram(model_size_gb=model_size_gb, gpu_vram_gb=gpu_vram_gb,
                         idle_used_gb=idle_used_gb,
                         precheck_enabled=vram_precheck)
            _append_log(job_id, L("tune.log.hw", vram=f"{gpu_vram_gb:.1f}", model=model,
                                  size=f"{model_size_gb:.1f}", ctx=ctx_size,
                                  goal=goal_label(goal)))
            all_results = []
            baseline_gpu_ok = False

            # 基线：baseline_cfg 是否为 None 表示「用户要不要基线」；
            # 要的话，优先用目标机上正在跑的 llama-server 命令行（issue #6 问题 6），
            # 因为那才是用户当前配置的真实来源 → 读不到才回落到前端传的内置基线。
            base_cfg = None
            if baseline_cfg is not None:
                live = _read_live_cfg(executor, target)
                if live:
                    base_cfg = live
                    _append_log(job_id, L(
                        "tune.log.baseline_live",
                        cfg=" ".join(f"{k}={v}" for k, v in base_cfg.items())))
                else:
                    base_cfg = baseline_cfg

            if base_cfg:
                _append_log(job_id, L("tune.log.baseline_test"))
                b = _run_one(executor, target, engine, model_path, base_cfg,
                             ctx_size, job_id, "baseline", vram)
                if b:
                    b["score"] = _score(b["metrics"], goal)
                    all_results.append(b)
                    with _LOCK:
                        _JOBS[job_id]["baseline"] = b
                    _append_log(job_id, L("tune.log.baseline_score", score=b['score']))
                    # 基线在 GPU 上真跑通了 → 用两点差分反推 KV 字节/token
                    if b["metrics"].get("decode", 0) > 0 \
                            and base_cfg.get("n-gpu-layers") != "0":
                        baseline_gpu_ok = True
                        cache_k = base_cfg.get("cache-type-k", "f16")
                        # 两点必须走同一条采集路径，只让 ctx 不同。
                        # 曾把基线那一点直接当低档复用，但基线跑过 3 遍 bench，
                        # bench 结束后的显存未必回到纯驻留水平；拿它当低档等于把
                        # 「bench 遗留差异」和「ctx 增长」混进同一个差值，斜率被系统性
                        # 压低——真机实测偏低 2.88 倍（90876 vs 理论 262144），
                        # 后果是 ctx≥32768 时把本该 OOM 的组合放行。
                        used_lo = _probe_used_gb(
                            executor, target, engine, model_path, base_cfg,
                            ctx_size, job_id, "calib-lo", vram)
                        # 再探一档更大的 ctx，只读显存不跑测速。两点相减能消掉
                        # MTP draft、compute buffer 这类**不随 ctx 增长**的固定开销，
                        # 而单点反推会把这些开销除以 ctx 错当成 KV 斜率
                        ctx_hi = ctx_size * CALIB_CTX_MULTIPLIER
                        used_hi = _probe_used_gb(
                            executor, target, engine, model_path, base_cfg,
                            ctx_hi, job_id, "calib", vram)
                        probes = []
                        if used_lo > 0:
                            probes.append((ctx_size, used_lo))
                        if used_hi > 0:
                            probes.append((ctx_hi, used_hi))
                        # 探针失败就放弃校准、回到 advisory 放行；绝不退回单点反推，
                        # 单点正是把固定开销错算成 KV 的根源（错误拦截比放行更糟）
                        if _calibrate(vram, probes, cache_k):
                            _append_log(job_id, L(
                                "tune.log.vram_calib",
                                kv=_fmt_num(vram.kv_bytes_per_token),
                                n=vram.calibrated_ctx, a=ctx_size, b=ctx_hi))
                        else:
                            _append_log(job_id, L("tune.log.vram_nocalib"))
                else:
                    _append_log(job_id, L("tune.log.baseline_fail"))

            # 阶段一 coarse
            coarse_best = _coarse_search(executor, target, engine, model_path,
                                         ctx_size, vram, goal, job_id,
                                         baseline_gpu_ok)
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
        # -L 跟随符号链接：模型常以软链形式放在 models 目录（如按量化版本切换），
        # 不加 -L 时 stat 返回链接本身的大小（几十字节），权重会被当成 0 GB。
        cmd = f'stat -Lc %s "{model_path}" 2>/dev/null || echo 0'
    r = executor.run(cmd, timeout=10)
    digits = "".join(c for c in r.stdout if c.isdigit())
    if digits:
        return round(int(digits) / (1024 ** 3), 1)
    return 0.0


def _get_gpu_vram(executor: Executor, target: Target) -> float:
    """单卡显存上限（GB）——多卡取**最大单卡**，与 _gpu_mem_used_gb 同口径。

    不直接信任 `_detect_gpu_static`：它按 `split(",")` 拆 nvidia-smi 的 CSV，
    而多卡时输出是多行，第二行的卡名会被当成 driver_version。tuner 的每个候选
    都只往一张卡上装，所以这里也只看单卡、取最大值，不做求和。
    """
    r = executor.run(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", timeout=8)
    totals = []
    for tok in r.stdout.split():
        try:
            totals.append(float(tok))
        except ValueError:
            continue
    if totals:
        return round(max(totals) / 1024, 2)
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
