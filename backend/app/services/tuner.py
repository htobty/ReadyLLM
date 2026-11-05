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

# KV cache 量化每元素字节数（同量化下 KV 占用与字节数成正比，用于跨量化换算）
_CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}


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


def _calibrate(vram: _Vram, used_with_server_gb: float, ctx_size: int,
               cache_type: str) -> bool:
    """用基线跑起来时的实测显存反推 KV 字节/token。

    减去的是"这台机器上不含本模型的占用"（idle_used），而不是 0：
    桌面环境、别的服务、以及上一次没退干净的进程都会算进去，直接减 0 会把
    别人的占用当成 KV cache，估出天文数字。
    """
    if used_with_server_gb <= 0 or ctx_size <= 0:
        return False
    kv_gb = used_with_server_gb - vram.idle_used_gb - vram.model_size_gb
    # 扣完是负数或接近 0：说明权重没全进显存、或实测值没意义，不做校准
    if kv_gb <= 0.05:
        return False
    cache_b = _CACHE_BYTES.get(cache_type, 2.0)
    vram.kv_bytes_per_token = (kv_gb * (1024 ** 3) / ctx_size) * (2.0 / cache_b)
    vram.calibrated_ctx = ctx_size
    vram.cache_bytes_at_calib = 2.0
    return True


def _fits_vram(cfg: dict, vram: _Vram, ctx_size: int) -> bool:
    """能否放进显存；n-gpu-layers=0 视为 CPU 兜底（慢但总能跑）。

    只有校准过才拦截。没校准 / 读不到显卡 / 权重大小未知时一律放行，
    把判定权交回给真实启动——这正是旧版出问题的地方：它会用一个错公式
    把可行配置全拦掉，然后静默降级到 CPU 产出无效结果。
    """
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
    if not _wait_ready(executor, target):
        _append_log(job_id, L("tune.log.run_timeout", tag=tag, label=label))
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
                if _fits_vram(cfg, vram, ctx_size):
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
            # 空载已用显存：既是"等显存释放"的目标水位，也是校准时该扣掉的底噪
            idle_used_gb = _gpu_mem_used_gb(executor, target)
            vram = _Vram(model_size_gb=model_size_gb, gpu_vram_gb=gpu_vram_gb,
                         idle_used_gb=idle_used_gb)
            _append_log(job_id, L("tune.log.hw", vram=f"{gpu_vram_gb:.1f}", model=model,
                                  size=f"{model_size_gb:.1f}", ctx=ctx_size,
                                  goal=goal_label(goal)))
            all_results = []
            baseline_gpu_ok = False

            # 基线：用户原始参数先测一遍
            if baseline_cfg:
                _append_log(job_id, L("tune.log.baseline_test"))
                b = _run_one(executor, target, engine, model_path, baseline_cfg,
                             ctx_size, job_id, "baseline", vram)
                if b:
                    b["score"] = _score(b["metrics"], goal)
                    all_results.append(b)
                    with _LOCK:
                        _JOBS[job_id]["baseline"] = b
                    _append_log(job_id, L("tune.log.baseline_score", score=b['score']))
                    # 基线在 GPU 上真跑通了 → 用它反推 KV 字节/token，预检才有依据
                    if b["metrics"].get("decode", 0) > 0 \
                            and baseline_cfg.get("n-gpu-layers") != "0":
                        baseline_gpu_ok = True
                        cache_k = baseline_cfg.get("cache-type-k", "f16")
                        if _calibrate(vram, b.get("gpu_mem_used_gb", 0.0),
                                      ctx_size, cache_k):
                            _append_log(job_id, L(
                                "tune.log.vram_calib",
                                kv=_fmt_num(vram.kv_bytes_per_token),
                                n=vram.calibrated_ctx))
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
