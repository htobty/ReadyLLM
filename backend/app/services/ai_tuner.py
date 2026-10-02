"""AI Agent 调优服务

Agent 模式：调用用户配置的大模型 API，让 LLM 基于硬件/模型/场景推理出
最优参数，并支持多轮迭代（测速结果喂回 → LLM 二次优化 → 再测 → ...）。

流程：
  1. 构造 system prompt（硬件、模型、参数白名单、JSON 格式要求）
  2. 调 LLM → 解析 action: test(给参数) / done(最终推荐)
  3. action=test → 启动模型测速 → 结果追加到对话 → 回到 2
  4. action=done → 输出最终推荐 + 分析
  5. 达到最大轮次强制结束

配置持久化到 ~/.model-deploy-assistant/ai_config.json
"""

import json
import os
import threading
import time
import uuid
import urllib.request
import urllib.error
from typing import Optional, List, Dict

from ..models.target import Target, get_target
from .executor import Executor
from .engine_adapter import StartParams
from .llama_cpp import LlamaCppAdapter
from .i18n import L, get_lang, set_lang
from .config_generator import generate_config

# ==================== 配置 ====================

_CONFIG_DIR = os.path.expanduser("~/.model-deploy-assistant")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "ai_config.json")

# 参数白名单：LLM 只能推荐这些参数
PARAM_WHITELIST = [
    "batch-size", "ubatch-size", "threads", "threads-batch",
    "n-gpu-layers", "gpu-layers", "gpu-layers-draft",
    "cache-type-k", "cache-type-v",
    "flash-attn", "spec-type", "spec-draft-n-max", "spec-draft-n-min",
    "spec-draft-p-min", "spec-draft-ngl", "fit", "parallel", "numa", "mlock", "no-mmap",
    "rope-scaling", "keep", "tensor-split", "main-gpu", "split-mode",
    "override-kv", "load-mode",
]

MAX_ROUNDS = 8  # 最大迭代轮次


def get_config() -> dict:
    """读取 AI 配置"""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"api_url": "", "api_key": "", "model_name": ""}


def save_config(cfg: dict):
    """保存 AI 配置"""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def test_connection(cfg: dict) -> dict:
    """测试 LLM API 连通性，返回 {ok, message, model_info}"""
    url = cfg.get("api_url", "").rstrip("/")
    if not url:
        return {"ok": False, "message": L("tune.ai.conn.empty_url")}
    # 尝试 /v1/models 端点
    models_url = f"{url}/models" if "/v1" in url else f"{url}/v1/models"
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    try:
        req = urllib.request.Request(models_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("id", "") for m in data.get("data", [])]
            return {"ok": True, "message": L("tune.ai.conn.ok", models=", ".join(models[:5])),
                    "models": models}
    except urllib.error.HTTPError as e:
        # 401/403 说明地址对但认证问题；404 可能没有 /models 端点但服务在
        if e.code in (401, 403):
            return {"ok": False, "message": L("tune.ai.conn.auth_fail", code=e.code)}
        return {"ok": True, "message": L("tune.ai.conn.reachable", code=e.code)}
    except Exception as e:
        return {"ok": False, "message": L("tune.ai.conn.fail", err=e)}


# ==================== Prompt 构造 ====================

def _build_system_prompt(hardware: dict, model_info: dict, ctx_size: int,
                         goal: str, user_desc: str,
                         baseline_params: dict = None,
                         baseline_metrics: dict = None) -> str:
    """构造 system prompt。
    新架构：LLM 的角色是"精调专家"，不是"从零猜参数"。
    确定性生成器已经给出经过计算验证的基础配置，LLM 在此基础上做小幅探索。
    """
    hw_lines = []
    gpu = hardware.get("gpu", {})
    cpu = hardware.get("cpu", {})
    mem = hardware.get("memory", {})
    if gpu:
        hw_lines.append(f"- GPU: {gpu.get('name', '未知')} {gpu.get('total_memory_gb', '?')}GB 显存")
    if cpu:
        hw_lines.append(f"- CPU: {cpu.get('name', '未知')} {cpu.get('cores', '?')}核{cpu.get('threads', '?')}线程")
    if mem:
        hw_lines.append(f"- 内存: {mem.get('total_gb', '?')}GB")
    hw_lines.append(f"- 系统: {hardware.get('os', '未知')}")

    mi_lines = [
        f"- 文件: {model_info.get('filename', '未知')}",
        f"- 大小: {model_info.get('size_gb', '?')}GB",
    ]

    # 基线信息（确定性生成器输出 + 实测结果）
    baseline_section = ""
    if baseline_params and baseline_metrics:
        baseline_section = f"""
## 已验证的基线配置（你的起点）
以下参数由确定性算法生成并已实测验证，是当前已知的最优起点：

参数: {json.dumps(baseline_params, ensure_ascii=False)}

实测结果:
- 解码速度: {baseline_metrics.get('decode', '?')} t/s
- 预填充速度: {baseline_metrics.get('prefill', '?')} t/s
- GPU 利用率: {baseline_metrics.get('gpu_util', '?')}%
- GPU 显存: {baseline_metrics.get('gpu_mem_pct', '?')}%
- CPU: {baseline_metrics.get('cpu_pct', '?')}%

你的任务是在这个基线上做小幅精调，尝试找到更好的配置。
不要大幅偏离基线（如把 ngl 改成部分卸载、去掉投机解码），这些已经被验证是最优方向。
"""

    # 英文界面的调优：LLM 的自由文本（reasoning）会原样渲染到界面与日志，而这份提示词
    # 和后续每轮的反馈消息都是中文，模型会顺口用中文作答。这里显式指定输出语言，并强调
    # 与输入语言无关——否则中文的测速反馈会把它带回中文。
    # 中文界面不注入任何内容，原有提示词保持逐字节不变。
    lang_rule = ""
    lang_tip = ""
    if get_lang() == "en":
        lang_rule = """
## Output language
Write all natural-language fields in English, especially `reasoning`. This rule holds
regardless of the language used in this prompt or in the feedback messages that follow:
those are written in Chinese, and your output must still be English.
JSON keys and parameter names stay unchanged.
"""
        lang_tip = "\n- `reasoning` must be written in English (not Chinese)"

    return f"""你是一个 llama.cpp 推理参数精调专家。系统已经通过确定性算法生成了一组经过验证的基础配置，你的任务是在此基础上做小幅探索，寻找可能的性能提升。{lang_rule}

## 硬件环境
{chr(10).join(hw_lines)}

## 模型信息
{chr(10).join(mi_lines)}

## 用户需求
- 上下文长度: {ctx_size}
- 优化目标: {goal}
- 场景描述: {user_desc or '未提供'}
{baseline_section}
## 可用参数白名单
你只能推荐以下参数（不要发明不存在的参数）：
{', '.join(PARAM_WHITELIST)}

## 不可违反的硬约束
1. n-gpu-layers 必须始终为 "all"。绝对不要尝试部分卸载。
2. 如果基线已启用投机解码（spec-type=draft-mtp），不要关闭它。投机解码是最大的速度杠杆。
3. 必须包含 --fit off
4. flash-attn 的值只能是 "on" 或 "off"
5. 显存不足时降级顺序：f16 → q8_0 → q4_0（降 cache 量化），绝不减 GPU 层数
6. ctx-size 由用户固定，禁止在 params 中输出或修改 ctx-size（即使显存不足也不能动它，要降就降 cache 量化）

## 你可以探索的方向（按优先级）
1. spec-draft-n-max: 尝试 2/3/4/5（影响投机解码接受长度）
2. spec-draft-p-min: 尝试 0/0.5/0.8（投机接受概率下限：draft 头一旦不够确定就提前收手，
   不再往下瞎猜。调高它可以让更大的 n-max 更划算——见 llama.cpp discussion #25198）
3. batch-size / ubatch-size: 在基线附近 ±50% 范围微调
4. cache-type-k/v: 如果基线用 f16，可试 q8_0 看是否有速度差异（通常差异 <5%）
5. threads: 在物理核数附近 ±2 微调
6. parallel: 如果有并发需求可尝试 2
7. 如果基线没有启用投机解码（显存不够），不要强行启用

## 输出格式（严格 JSON）
每轮你必须输出一个 JSON 对象：

如果要测试一组参数：
{{"action": "test", "params": {{"参数名": "值", ...}}, "reasoning": "为什么选这组参数的简要分析"}}

如果认为已经找到最优或无法继续优化：
{{"action": "done", "params": {{"参数名": "值", ...}}, "reasoning": "最终推荐的分析说明", "confidence": "high/medium/low"}}

注意：
- params 中的值全部用字符串
- 不要输出 JSON 以外的内容
- 每轮只输出一组参数
- 每次只改 1-2 个参数，不要同时改太多（否则无法判断哪个变化有效）{lang_tip}
"""


def _build_test_result_message(round_num: int, params: dict, metrics: dict) -> str:
    """构造测速结果反馈消息（含 CPU/内存指标，供 AI 综合分析）"""
    mem_info = ""
    if metrics.get("mem_total_gb"):
        mem_info = f"- 系统内存: {metrics.get('mem_used_gb', 0)}GB / {metrics.get('mem_total_gb', 0)}GB ({metrics.get('mem_pct', 0)}%)\n"
    return f"""第 {round_num} 轮测试结果：

测试参数: {json.dumps(params, ensure_ascii=False)}

【推理性能】
- 解码速度: {metrics.get('decode', 0)} t/s
- 预填充速度: {metrics.get('prefill', 0)} t/s
- 首字延迟(TTFT): {metrics.get('ttft_ms', 0)} ms

【GPU 状态】
- GPU 利用率: {metrics.get('gpu_util', 0)}%
- GPU 显存占用: {metrics.get('gpu_mem_pct', 0)}%

【CPU 与内存】
- CPU 利用率: {metrics.get('cpu_pct', 0)}%
{mem_info}
请综合以上所有指标分析：
- GPU 利用率低但显存占满 → 可能是 memory-bound（正常），不要因此降层数
- CPU 利用率过高 → 可能 threads 设置不合理或有层落在 CPU
- 内存占用接近总量 → 有 swap 风险，需降低 ctx 或 cache
- 如果还有优化空间，输出 action=test 和新的参数组合
- 如果已经最优或无法继续改善，输出 action=done 和最终推荐
"""


# ==================== LLM 调用 ====================

def _call_llm(cfg: dict, messages: List[dict], job_id: str = None) -> Optional[str]:
    """调用 OpenAI 兼容 API，返回 assistant 消息内容。
    失败时把真实原因写入 job 日志（之前 except Exception 直接吞掉，无法定位）。
    """
    def _log(msg: str):
        if job_id:
            _append_log(job_id, msg)

    url = cfg.get("api_url", "").rstrip("/")
    if not url:
        _log(L("tune.ai.llm.no_url"))
        return None
    if not url.endswith("/chat/completions"):
        if "/v1" in url:
            url = f"{url}/chat/completions"
        else:
            url = f"{url}/v1/chat/completions"
    _log(L("tune.ai.llm.request", url=url, model=cfg.get("model_name", "")))

    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    else:
        _log(L("tune.ai.llm.no_key"))

    payload = json.dumps({
        "model": cfg.get("model_name", ""),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            # 请求成功但无 choices：多半是模型名不对或返回结构异常
            _log(L("tune.ai.llm.no_choices", raw=json.dumps(data, ensure_ascii=False)[:400]))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:400]
        except Exception:
            pass
        _log(L("tune.ai.llm.http_error", code=e.code, body=body))
    except urllib.error.URLError as e:
        _log(L("tune.ai.llm.net_error", err=e.reason))
    except Exception as e:
        _log(L("tune.ai.llm.exception", err=f"{type(e).__name__}: {e}"))
    return None


def _parse_llm_response(text: str) -> Optional[dict]:
    """解析 LLM 返回的 JSON，容错处理"""
    if not text:
        return None
    # 尝试直接解析
    text = text.strip()
    # 去掉可能的 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找 JSON 子串
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _validate_params(params: dict) -> dict:
    """过滤白名单外的参数，返回合法子集；规范化布尔值"""
    valid = {}
    for k, v in params.items():
        if k in PARAM_WHITELIST:
            valid[k] = str(v)
    # 规范化：flash-attn 只认 on/off，不认 true/false
    if "flash-attn" in valid:
        fa = valid["flash-attn"].lower()
        if fa in ("true", "1", "yes", "enabled"):
            valid["flash-attn"] = "on"
        elif fa in ("false", "0", "no", "disabled"):
            valid["flash-attn"] = "off"
    # ctx-size 是用户固定的约束，绝不允许 AI 改动（防御 AI 仍返回它）
    valid.pop("ctx-size", None)
    # 强制加入必要参数
    valid["fit"] = "off"
    valid["metrics"] = ""
    valid["host"] = "0.0.0.0"
    return valid


# ==================== Agent 任务 ====================

_JOBS: dict = {}
_LOCK = threading.Lock()


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
    """返回该目标机正在运行的 AI 调优任务摘要，供前端刷新后恢复轮询。"""
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
                "round_count": len(job.get("rounds", [])),
            })
        return out


def start_ai_tune(target_id: str, model: str, ctx_size: int,
                  goal: str, user_desc: str) -> dict:
    """启动 AI Agent 调优任务"""
    # 后台线程不继承请求上下文，先捕获发起时的界面语言，进 worker 再设回去
    lang = get_lang()
    target = get_target(target_id)
    if not target:
        return {"ok": False, "message": L("tune.err.no_target")}
    if not target.engine_path:
        return {"ok": False, "message": L("tune.ai.err.no_engine")}
    if getattr(target, "engine_type", "llama_cpp") != "llama_cpp":
        return {"ok": False, "message": L("tune.ai.err.llama_only")}

    cfg = get_config()
    if not cfg.get("api_url"):
        return {"ok": False, "message": L("tune.ai.err.no_api")}

    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id, "target_id": target_id, "model": model,
            "ctx_size": ctx_size, "goal": goal,
            "status": "running", "logs": [], "rounds": [],
            "best": None, "error": "",
        }

    def _worker():
        executor = None
        set_lang(lang)
        try:
            from .executor import make_executor
            from .collectors import path_join, detect_hardware
            executor = make_executor(target)
            engine = LlamaCppAdapter(executor, target)

            if not engine.check_installed():
                _fail(job_id, L("tune.ai.log.no_engine"))
                return

            # 采集硬件信息
            _append_log(job_id, L("tune.ai.log.collect_hw"))
            hardware = detect_hardware(executor, target)
            hardware["os"] = target.os

            # 模型信息
            model_path = path_join(target, target.models_dir, model)
            model_size_gb = _get_model_size(executor, target, model_path)
            model_info = {"filename": model, "size_gb": model_size_gb}

            _append_log(job_id, L("tune.ai.log.hw",
                                  gpu=hardware.get("gpu", {}).get("name", "?"),
                                  model=model, size=model_size_gb, ctx=ctx_size))

            # ===== 新架构：确定性生成器出基线 → 实测 → 喂给 LLM =====
            gpu_info = hardware.get("gpu", {})
            cpu_info = hardware.get("cpu", {})
            gpu_vram = gpu_info.get("total_memory_gb", 8)
            cpu_cores = cpu_info.get("cores", 8)
            cpu_threads = cpu_info.get("threads", 16)

            # 基线来源优先级：上次调优结果 > 确定性生成器
            from .tune_history import get_latest as _hist_get
            _hist = _hist_get(target_id, model)
            if _hist and _hist.get("params"):
                baseline_params = dict(_hist["params"])
                _src = L("tune.src.tuner") if _hist.get("source") == "tuner" else L("tune.src.ai")
                _append_log(job_id, L("tune.ai.log.baseline_from_history", src=_src,
                                      score=_hist.get("score", 0), ts=_hist.get("ts", "")))
                _append_log(job_id, L("tune.ai.log.params",
                                      params=json.dumps(baseline_params, ensure_ascii=False)))
            else:
                _append_log(job_id, L("tune.ai.log.gen_config"))
                gen_result = generate_config(
                    gpu_vram_gb=gpu_vram,
                    model_size_gb=model_size_gb,
                    model_filename=model,
                    ctx_size=ctx_size,
                    cpu_cores=cpu_cores,
                    cpu_threads=cpu_threads,
                )
                baseline_params = gen_result["params"]
                for r in gen_result["reasoning"]:
                    _append_log(job_id, f"  · {r}")
                for w in gen_result.get("warnings", []):
                    _append_log(job_id, f"  ⚠ {w}")

            # 实测基线配置
            _append_log(job_id, L("tune.ai.log.baseline_test"))
            valid_baseline = _validate_params(baseline_params)
            _append_log(job_id, L("tune.ai.log.params",
                                  params=json.dumps(valid_baseline, ensure_ascii=False)))
            baseline_metrics = _run_test(executor, target, engine, model_path,
                                         valid_baseline, ctx_size, job_id)

            if baseline_metrics:
                _append_log(job_id, L("tune.ai.log.baseline_ok",
                                      decode=baseline_metrics.get('decode', 0),
                                      prefill=baseline_metrics.get('prefill', 0),
                                      gpu=baseline_metrics.get('gpu_util', 0)))
            else:
                _append_log(job_id, L("tune.ai.log.baseline_fail"))

            # 记录基线为第 0 轮
            with _LOCK:
                _JOBS[job_id]["rounds"].append({
                    "round": 0,
                    "params": valid_baseline,
                    "metrics": baseline_metrics,
                    "reasoning": L("tune.reason.deterministic"),
                })

            # 构造 LLM 对话（含基线信息）
            system_prompt = _build_system_prompt(
                hardware, model_info, ctx_size, goal, user_desc,
                baseline_params=valid_baseline,
                baseline_metrics=baseline_metrics,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "基线已实测完毕，请在此基础上探索可能的性能提升。每次只改 1-2 个参数。"},
            ]

            best_result = None
            best_score = baseline_metrics.get("decode", 0) if baseline_metrics else -1
            if baseline_metrics:
                best_result = {"params": valid_baseline, "metrics": baseline_metrics, "round": 0}

            for round_num in range(1, MAX_ROUNDS + 1):
                _append_log(job_id, L("tune.ai.log.round", n=round_num, max=MAX_ROUNDS))

                # 调 LLM
                response = _call_llm(cfg, messages, job_id)
                if response is None:
                    _fail(job_id, L("tune.fail.round_llm", n=round_num))
                    return

                parsed = _parse_llm_response(response)
                if parsed is None:
                    _append_log(job_id, L("tune.ai.log.parse_fail", raw=response[:200]))
                    # 把错误反馈给 LLM 重试
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "你的输出不是合法 JSON，请严格按格式重新输出。"})
                    continue

                action = parsed.get("action", "")
                reasoning = parsed.get("reasoning", "")
                params = parsed.get("params", {})

                _append_log(job_id, L("tune.ai.log.ai_reasoning", text=reasoning[:150]))

                if action == "done":
                    _append_log(job_id, L("tune.ai.log.ai_done",
                                          conf=parsed.get("confidence", "?")))
                    final_params = _validate_params(params)
                    with _LOCK:
                        job = _JOBS[job_id]
                        job["best"] = {
                            "params": final_params,
                            "reasoning": reasoning,
                            "confidence": parsed.get("confidence", "medium"),
                            "round": round_num,
                        }
                        job["status"] = "success"
                        _tid, _model, _ctx = job["target_id"], job["model"], job["ctx_size"]
                    _append_log(job_id, L("tune.ai.log.ai_final",
                                          params=json.dumps(final_params, ensure_ascii=False)))
                    # 落盘最近调优参数，供部署页作为默认参数回填
                    try:
                        from .tune_history import save_latest
                        save_latest(_tid, _model, _ctx, final_params,
                                    source="ai_tuner", score=best_score)
                    except Exception as e:
                        _append_log(job_id, L("tune.log.save_fail", err=e))
                    return

                if action != "test":
                    _append_log(job_id, L("tune.ai.log.unknown_action", action=action))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "action 必须是 test 或 done，请重新输出。"})
                    continue

                # 执行测试
                valid_params = _validate_params(params)
                _append_log(job_id, L("tune.ai.log.test_params",
                                      params=json.dumps(valid_params, ensure_ascii=False)))

                metrics = _run_test(executor, target, engine, model_path,
                                    valid_params, ctx_size, job_id)

                if metrics is None:
                    _append_log(job_id, L("tune.ai.log.test_fail"))
                    test_msg = f"第 {round_num} 轮测试失败：模型启动超时或参数无效。请换一组参数重试。"
                else:
                    score = metrics.get("decode", 0)
                    _append_log(job_id, L("tune.ai.log.result",
                                          decode=metrics['decode'], prefill=metrics['prefill'],
                                          gpu=metrics['gpu_util'], vram=metrics['gpu_mem_pct'],
                                          cpu=metrics.get('cpu_pct', 0),
                                          mem=metrics.get('mem_used_gb', 0),
                                          memtotal=metrics.get('mem_total_gb', 0)))
                    test_msg = _build_test_result_message(round_num, valid_params, metrics)

                    # 记录最佳
                    if score > best_score:
                        best_score = score
                        best_result = {"params": valid_params, "metrics": metrics, "round": round_num}

                # 记录本轮
                with _LOCK:
                    _JOBS[job_id]["rounds"].append({
                        "round": round_num,
                        "params": valid_params,
                        "metrics": metrics,
                        "reasoning": reasoning,
                    })

                # 喂回结果
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": test_msg})

            # 达到最大轮次
            _append_log(job_id, L("tune.ai.log.max_rounds", max=MAX_ROUNDS))
            if best_result:
                with _LOCK:
                    job = _JOBS[job_id]
                    job["best"] = {
                        "params": best_result["params"],
                        "reasoning": L("tune.reason.max_rounds", n=best_result['round']),
                        "confidence": "medium",
                        "round": best_result["round"],
                    }
                    job["status"] = "success"
            else:
                _fail(job_id, L("tune.fail.all_rounds"))

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


def _run_test(executor: Executor, target: Target, engine: LlamaCppAdapter,
              model_path: str, params: dict, ctx_size: int, job_id: str) -> Optional[dict]:
    """启动模型 → 测速 → 停止，返回 metrics 或 None"""
    from .tuner import _wait_ready, _bench_median

    engine.stop()
    time.sleep(2)

    # 构造参数列表（ctx-size/port/metrics/host 由下面统一追加，避免重复）
    args = []
    for k, v in params.items():
        if k in ("metrics", "host", "ctx-size", "port"):
            continue
        if v == "":
            args.append(f"--{k}")
        else:
            args += [f"--{k}", str(v)]
    args += [
        "--ctx-size", str(ctx_size),
        "--metrics",
        "--host", "0.0.0.0",
        "--port", str(target.service_port),
    ]

    ok, msg = engine.start(StartParams(model_path=model_path, extra_args=args))
    if not ok:
        _append_log(job_id, L("tune.ai.log.start_fail", err=msg))
        return None

    if not _wait_ready(executor, target):
        _append_log(job_id, L("tune.ai.log.start_timeout"))
        engine.stop()
        return None

    metrics = _bench_median(executor, target, ctx_size)
    engine.stop()
    time.sleep(2)
    return metrics


def _get_model_size(executor: Executor, target: Target, model_path: str) -> float:
    """探测模型文件大小 GB"""
    if target.os == "windows":
        cmd = (f'powershell -Command "if(Test-Path \'{model_path}\')'
               f'{{(Get-Item \'{model_path}\').Length}}else{{0}}"')
    else:
        cmd = f'stat -c %s "{model_path}" 2>/dev/null || echo 0'
    r = executor.run(cmd, timeout=10)
    digits = "".join(c for c in r.stdout if c.isdigit())
    if digits:
        return round(int(digits) / (1024 ** 3), 1)
    return 0.0


def _fail(job_id: str, err: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = err
    _append_log(job_id, f"✗ {err}")
