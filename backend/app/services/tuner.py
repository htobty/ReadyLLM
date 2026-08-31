"""智能调优压测服务（两阶段搜索版）

在用户当前硬件配置下，为目标机上的某个模型搜索最优推理参数。

设计要点：
  1. 参数分三类：
     - 约束类（用户/硬件定死）：模型、端口、ctx 下限 → 不搜，作硬约束
     - 离散高影响类：spec-type / cache-type-k,v / n-gpu-layers → coarse 阶段搜
     - 连续微调类：batch-size / ubatch-size / spec-draft-n-max → fine 阶段坐标下降
  2. 显存可行性预检：估算权重+KV+cache 占用，放不下的组合直接跳过，不浪费启动时间
  3. 两阶段搜索：coarse 定主导因素 → fine 在最优组合附近收敛
  4. 可信测速：warmup + 正式 3 次取中位数，记录解码/预填充/TTFT/GPU 利用率
  5. 基线对比：以用户原始参数为 baseline 先测，输出"推荐 vs 当前"

优化目标可选：latency(端到端体感,默认) / throughput(纯解码吞吐) / prefill(长文本预填充)。
无引擎/无模型时明确报错，绝不返回模拟数据。
"""

import threading
import time
import json
import uuid
from statistics import median
from typing import Optional, List, Dict

from .executor import Executor
from .engine_adapter import StartParams
from .llama_cpp import LlamaCppAdapter
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
_BENCH_MAX_TOKENS = 128
_BENCH_REPEATS = 3  # 正式测速重复次数，取中位数

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
}

# 目标可选评分权重：(解码速度, 预填充速度, TTFT)
GOAL_WEIGHTS = {
    "latency":    {"decode": 0.5, "prefill": 0.3, "ttft": 0.2},
    "throughput": {"decode": 1.0, "prefill": 0.0, "ttft": 0.0},
    "prefill":    {"decode": 0.2, "prefill": 0.8, "ttft": 0.0},
}
GOAL_LABELS = {
    "latency": "端到端体感",
    "throughput": "解码吞吐",
    "prefill": "长文本预填充",
}


def _normalize_cfg(spec_type, cache_type, ngl, batch, ubatch, draft_n_max) -> dict:
    """一个完整配置 = 离散主导因素 + 连续微调参数"""
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
    return cfg


def _cfg_label(cfg: dict) -> str:
    return " / ".join(f"{k}={v}" for k, v in cfg.items())


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


def _estimate_vram_gb(model_size_gb: float, ctx_size: int,
                      cache_type: str, kv_heads_dim: int = 8192) -> float:
    """粗估显存占用 GB：权重 + KV cache。
    kv_heads_dim 为 KV 维度近似（hidden*n_heads 量级），27B 级约 8192。
    权重全进 GPU（n-gpu-layers=all 场景）；CPU 兜底场景由调用方单独处理。
    """
    # KV cache: 2(K+V) * ctx * kv_dim * bytes * 层数比例近似
    # 简化：ctx * kv_dim * cache_bytes * 2 / 1e9，再乘层数经验系数
    kv_bytes = 2 * ctx_size * kv_heads_dim * _CACHE_BYTES.get(cache_type, 2.0)
    # 27B 约 64 层，每层都有 KV；上面 2* 已含 K/V，这里再乘层数
    kv_gb = kv_bytes * 64 / (1024 ** 3)
    return model_size_gb + kv_gb


def _fits_vram(cfg: dict, model_size_gb: float, ctx_size: int, gpu_vram_gb: float) -> bool:
    """判断配置是否放得进显存；n-gpu-layers=0 视为 CPU 兜底，总能'放下'（慢）"""
    if cfg.get("n-gpu-layers") == "0":
        return True
    est = _estimate_vram_gb(model_size_gb, ctx_size, cfg.get("cache-type-k", "f16"))
    # 留 10% 余量给激活值/显存碎片
    return est <= gpu_vram_gb * 0.9


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
    """warmup 1 次 + 正式 _BENCH_REPEATS 次，各指标取中位数"""
    _bench_once(executor, target, ctx_size)  # warmup，丢弃
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
    engine.stop()
    time.sleep(2)
    params = StartParams(model_path=model_path, extra_args=_args_list(cfg, target, ctx_size))
    ok, msg = engine.start(params)
    if not ok:
        _append_log(job_id, f"  [{tag}] {label} 启动失败: {msg}")
        return None
    if not _wait_ready(executor, target):
        _append_log(job_id, f"  [{tag}] {label} 启动超时(可能显存不足)")
        engine.stop()
        return None
    metrics = _bench_median(executor, target, ctx_size)
    engine.stop()
    time.sleep(2)
    _append_log(job_id, f"  [{tag}] {label} → 解码{metrics['decode']} t/s, "
                        f"预填充{metrics['prefill']} t/s, GPU {metrics['gpu_util']}%")
    return {"config": cfg, "label": label, "metrics": metrics}


def _coarse_search(executor, target, engine, model_path, ctx_size,
                   model_size_gb, gpu_vram_gb, goal, job_id) -> Optional[dict]:
    """阶段一：搜离散主导因素 spec-type × cache-type。
    n-gpu-layers 默认 all（全进 GPU）；仅当 all 全部超显存时才降级用 0 兜底，
    避免把纯 CPU 跑大模型这种必然慢的组合当常规候选浪费测速时间。"""
    def _build(ngl):
        out = []
        for spec in SPEC_OPTIONS:
            for cache in CACHE_OPTIONS:
                cfg = _normalize_cfg(spec, cache, ngl,
                                     CONTINUOUS_GRID["batch-size"][1],
                                     CONTINUOUS_GRID["ubatch-size"][1], 3)
                if _fits_vram(cfg, model_size_gb, ctx_size, gpu_vram_gb):
                    out.append(cfg)
                else:
                    _append_log(job_id, f"  跳过(显存不足): {_cfg_label(cfg)}")
        return out

    candidates = _build("all")
    if not candidates:
        _append_log(job_id, "  全 GPU 组合均超显存，降级用 CPU 兜底(n-gpu-layers=0)")
        candidates = _build("0")

    _append_log(job_id, f"【阶段1 coarse】{len(candidates)} 组主导因素组合")
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
    _append_log(job_id, f"  coarse 最优: {best['label']} (分 {best['score']})")
    return best


def _fine_search(executor, target, engine, model_path, ctx_size,
                 model_size_gb, gpu_vram_gb, goal, job_id,
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
        tune_params += ["spec-draft-n-max", "spec-draft-n-min"]

    _append_log(job_id, f"【阶段2 fine】坐标下降，调 {tune_params}")
    for param in tune_params:
        options = CONTINUOUS_GRID.get(param, [])
        improved = True
        while improved:
            improved = False
            cur_val = int(best["config"].get(param, options[0]))
            idx = options.index(cur_val) if cur_val in options else 0
            # 向两侧各探一步
            for ni in (idx - 1, idx + 1):
                if ni < 0 or ni >= len(options):
                    continue
                trial = dict(best["config"])
                trial[param] = str(options[ni])
                if not _fits_vram(trial, model_size_gb, ctx_size, gpu_vram_gb):
                    continue
                r = _run_one(executor, target, engine, model_path, trial, ctx_size,
                             job_id, f"fine {param}={options[ni]}")
                if r is None:
                    continue
                r["score"] = _score(r["metrics"], goal)
                if r["score"] > best["score"]:
                    best = r
                    improved = True
                    _append_log(job_id, f"    ✓ 改善: {param}={options[ni]} 分→{r['score']}")
                    break
    _append_log(job_id, f"  fine 收敛: {best['label']} (分 {best['score']})")
    return best


# ==================== 主流程 ====================

def start_tune(target_id: str, model: str, ctx_size: int = 8192,
               goal: str = "latency", baseline_cfg: Optional[dict] = None,
               model_size_gb: float = 0.0) -> dict:
    """启动两阶段调优任务。
    baseline_cfg：用户原始参数（dict），作为基线先测一组对比。
    model_size_gb：模型大小，用于显存预检；缺省按 0 跳过预检。
    """
    target = get_target(target_id)
    if not target:
        return {"ok": False, "message": "目标机器不存在"}
    if not target.engine_path:
        return {"ok": False, "message": "未配置推理引擎，请先在设置中安装"}
    if getattr(target, "engine_type", "llama_cpp") != "llama_cpp":
        return {"ok": False, "message": "自动调优目前仅支持 llama.cpp 引擎（vLLM 参数体系不同，暂不支持）"}
    if not target.models_dir or not model:
        return {"ok": False, "message": "未选择模型或模型目录为空"}
    if ctx_size < 1024:
        return {"ok": False, "message": "ctx-size 过小，请至少 1024"}

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
        try:
            from .executor import make_executor
            from .collectors import path_join
            executor = make_executor(target)
            engine = LlamaCppAdapter(executor, target)

            if not engine.check_installed():
                _fail(job_id, "目标机未检测到推理引擎，请先一键安装")
                _append_log(job_id, "✗ 未检测到推理引擎")
                return

            model_path = path_join(target, target.models_dir, model)

            # 取目标机显存；模型大小未传则自动探测（用于显存预检）
            gpu_vram_gb = _get_gpu_vram(executor, target)
            if model_size_gb <= 0:
                model_size_gb = _get_model_size_gb(executor, target, model_path)
            _append_log(job_id, f"目标机显存: {gpu_vram_gb:.1f} GB | 模型: {model} "
                                f"({model_size_gb:.1f} GB) | ctx 固定 {ctx_size} | 目标: "
                                f"{GOAL_LABELS.get(goal, goal)}")
            all_results = []

            # 基线：用户原始参数先测一遍
            if baseline_cfg:
                _append_log(job_id, "【基线】测试你当前配置")
                b = _run_one(executor, target, engine, model_path, baseline_cfg,
                             ctx_size, job_id, "baseline")
                if b:
                    b["score"] = _score(b["metrics"], goal)
                    all_results.append(b)
                    with _LOCK:
                        _JOBS[job_id]["baseline"] = b
                    _append_log(job_id, f"  基线分: {b['score']}")

            # 阶段一 coarse
            coarse_best = _coarse_search(executor, target, engine, model_path,
                                         ctx_size, model_size_gb, gpu_vram_gb,
                                         goal, job_id)
            if not coarse_best:
                _fail(job_id, "coarse 阶段无可用配置（可能显存不足）")
                with _LOCK:
                    _JOBS[job_id]["results"] = all_results
                return
            all_results.append(coarse_best)

            # 阶段二 fine
            fine_best = _fine_search(executor, target, engine, model_path,
                                     ctx_size, model_size_gb, gpu_vram_gb,
                                     goal, job_id, coarse_best["config"])
            all_results.append(fine_best)

            # 最终推荐 = fine 收敛结果；与基线对比
            final_best = fine_best if fine_best["score"] > 0 else coarse_best
            _finalize(job_id, all_results, final_best)
        except Exception as e:
            _fail(job_id, str(e))
            _append_log(job_id, f"✗ 异常: {e}")
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
        cmd = f'stat -c %s "{model_path}" 2>/dev/null || echo 0'
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
    _append_log(job_id, f"✓ 调优完成，推荐: {best['label']} (分 {best['score']})")
    # 落盘最近调优参数，供部署页作为默认参数回填
    try:
        from .tune_history import save_latest
        save_latest(_tid, _model, _ctx, best.get("config", {}),
                    source="tuner", score=best.get("score", 0))
    except Exception as e:
        _append_log(job_id, f"  ⚠ 调优结果落盘失败: {e}")
