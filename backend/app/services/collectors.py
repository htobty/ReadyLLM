"""硬件与监控数据采集器

所有采集基于用户配置的 Target 执行，按目标 OS（windows/macos/linux）适配命令，
不依赖任何硬编码环境。
"""

import json

from .executor import Executor
from ..models.target import Target


def path_join(target: Target, *parts: str) -> str:
    """按目标 OS 拼接路径"""
    sep = "\\" if target.os == "windows" else "/"
    base = parts[0].rstrip("\\/")
    return base + sep + sep.join(p.strip("\\/") for p in parts[1:])


def _parse_kv_lines(text: str) -> dict:
    """解析 key=value 形式输出"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _ps(body: str) -> str:
    """包装为 powershell -NoProfile -Command "..."。
    新版 Windows 已移除 wmic，统一改用 Get-CimInstance（Win8+/PS3+ 可用）。
    body 内只用单引号与 $ 变量，避免与外层双引号冲突。"""
    return f'powershell -NoProfile -Command "{body}"'


# ==================== GPU ====================

# 显示适配器注册表类（{4d36e968-...}）下每个实例的 HardwareInformation.qwMemorySize
# 是 64 位显存大小；Win32_VideoController.AdapterRAM 是 uint32，显存 >4GB 会回绕，
# 所以显存一律优先读注册表，AdapterRAM 只作兜底。
_WIN_DISPLAY_CLASS = ("HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
                      "{4d36e968-e325-11ce-bfc1-08002be10318}")


def _parse_mem_bytes(raw) -> float:
    """显存字节数 -> GB；不合理（回绕、非数字）的值返回 0"""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0 or val > 1024 ** 4:   # 上限 1TB，超出必是 uint32 回绕
        return 0.0
    return round(val / 1024 ** 3, 1)


def _normalize_gpu_vendor(compat: str) -> str:
    """WMI AdapterCompatibility -> amd / intel / nvidia / other"""
    c = (compat or "").lower()
    if "advanced micro devices" in c or "amd" in c or "ati " in c:
        return "amd"
    if "intel" in c:
        return "intel"
    if "nvidia" in c:
        return "nvidia"
    return "other"


def _collect_gpu(executor: Executor, target: Target) -> dict:
    """实时 GPU 采集：NVIDIA 用 nvidia-smi；Windows 上其余厂商走 GPU 性能计数器。
    Apple Silicon 无独立显存且 powermetrics 需 root，返回空。"""
    cmd = ("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,"
           "temperature.gpu,power.draw --format=csv,noheader,nounits")
    result = executor.run(cmd, timeout=8)
    if result.ok and result.stdout:
        parts = [p.strip() for p in result.stdout.split(",")]
        if len(parts) >= 6:
            name, util, mem_used, mem_total, temp, power = parts[:6]
            try:
                mem_used_f, mem_total_f = float(mem_used), float(mem_total)
                return {
                    "name": name,
                    "utilization": float(util),
                    "memory_used_gb": round(mem_used_f / 1024, 1),
                    "memory_total_gb": round(mem_total_f / 1024, 1),
                    "memory_pct": round(mem_used_f / mem_total_f * 100, 1) if mem_total_f > 0 else 0,
                    "temperature": int(float(temp)),
                    "power": float(power),
                }
            except ValueError:
                pass
    if target.os == "windows":
        return _collect_gpu_windows_generic(executor, target)
    return {}


def _collect_gpu_windows_generic(executor: Executor, target: Target) -> dict:
    """Windows 非 NVIDIA 显卡（AMD/Intel）的实时采集。

    nvidia-smi 只覆盖 NVIDIA，其余厂商没有等价的命令行工具，改走 Windows
    性能计数器（GPU Engine / GPU Adapter Memory，Win10 1809+ 提供）。走 WMI 类
    而不是 Get-Counter 的路径字符串：类名与属性名不随系统语言本地化，中文系统
    同样可用。温度与功耗没有通用接口，无从获取，只能留空。
    """
    result = executor.run(_ps(
        "$a=Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -notmatch 'Basic Display|Remote Display|Virtual' } | "
        "Sort-Object AdapterRAM -Descending | Select-Object -First 1; "
        "Write-Output ('NAME=' + $a.Name); "
        f"$m=(Get-ChildItem '{_WIN_DISPLAY_CLASS}' | "
        "ForEach-Object { (Get-ItemProperty $_.PSPath -Name "
        "'HardwareInformation.qwMemorySize' -ErrorAction SilentlyContinue)"
        ".'HardwareInformation.qwMemorySize' } | Measure-Object -Maximum).Maximum; "
        "if (-not $m) { $m=$a.AdapterRAM }; "
        "Write-Output ('TOTAL=' + $m); "
        "$u=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine "
        "-ErrorAction SilentlyContinue | Measure-Object UtilizationPercentage -Sum; "
        "Write-Output ('UTIL=' + $u.Sum); "
        "$d=Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
        "-ErrorAction SilentlyContinue | Measure-Object DedicatedUsage -Sum; "
        "Write-Output ('MEMUSED=' + $d.Sum)"
    ), timeout=20)
    kv = _parse_kv_lines(result.stdout)
    name = kv.get("NAME", "").strip()
    if not name:
        return {}
    total_gb = _parse_mem_bytes(kv.get("TOTAL", ""))
    out = {"name": name, "memory_total_gb": total_gb}
    # 计数器类不可用时这些项会读出空值：宁可不写（界面显示 "--"），
    # 也不要填 0 —— 0% 是有含义的值（GPU 真空闲），与"没读到"必须区分开
    util_raw = (kv.get("UTIL") or "").strip()
    if util_raw:
        try:
            # 各引擎利用率之和是近似值（与任务管理器口径接近），封顶 100
            out["utilization"] = min(round(float(util_raw), 1), 100.0)
        except ValueError:
            pass
    used_raw = (kv.get("MEMUSED") or "").strip()
    if total_gb > 0 and used_raw:
        try:
            used_gb = round(float(used_raw) / 1024 ** 3, 1)
            out["memory_used_gb"] = used_gb
            out["memory_pct"] = min(round(used_gb / total_gb * 100, 1), 100.0)
        except ValueError:
            pass
    return out


# ==================== CPU / 内存 ====================

def _collect_cpu_mem(executor: Executor, target: Target) -> dict:
    if target.os == "windows":
        cmd = _ps(
            "$c=(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average; "
            "$o=Get-CimInstance Win32_OperatingSystem; "
            "Write-Output ('LoadPercentage=' + $c); "
            "Write-Output ('TotalVisibleMemorySize=' + $o.TotalVisibleMemorySize); "
            "Write-Output ('FreePhysicalMemory=' + $o.FreePhysicalMemory)"
        )
        result = executor.run(cmd, timeout=20)
        kv = _parse_kv_lines(result.stdout)
        try:
            total = float(kv.get("TotalVisibleMemorySize", 0))
            free = float(kv.get("FreePhysicalMemory", 0))
            if total > 0:
                return {
                    "cpu_pct": float(kv.get("LoadPercentage", 0)),
                    "memory_used_gb": round((total - free) / 1024 / 1024, 1),
                    "memory_total_gb": round(total / 1024 / 1024, 1),
                    "memory_pct": round((total - free) / total * 100, 1),
                }
        except ValueError:
            pass
        return {}

    elif target.os == "macos":
        # shell 只输出原始数据，Python 端正则解析，避开 awk 引号问题
        import re
        cmd = (
            "top -l 1 | grep 'CPU usage'; "
            "vm_stat; "
            "echo MEMTOTAL=$(sysctl -n hw.memsize)"
        )
        result = executor.run(cmd, timeout=15)
        text = result.stdout
        try:
            idle = 100.0
            page_size = 4096
            active = wired = compressed = 0
            total = 0.0
            for line in text.splitlines():
                if "CPU usage" in line and "idle" in line:
                    nums = re.findall(r"([0-9.]+)%", line)
                    if nums:
                        idle = float(nums[-1])
                elif "page size of" in line:
                    m = re.findall(r"(\d+)", line)
                    if m:
                        page_size = int(m[0])
                elif line.startswith("Pages active"):
                    m = re.findall(r"(\d+)", line)
                    if m:
                        active = int(m[-1])
                elif line.startswith("Pages wired down"):
                    m = re.findall(r"(\d+)", line)
                    if m:
                        wired = int(m[-1])
                elif line.startswith("Pages occupied by compressor"):
                    m = re.findall(r"(\d+)", line)
                    if m:
                        compressed = int(m[-1])
                elif line.startswith("MEMTOTAL="):
                    m = re.findall(r"(\d+)", line)
                    if m:
                        total = float(m[-1])
            if total > 0:
                used_bytes = (active + wired + compressed) * page_size
                return {
                    "cpu_pct": round(100 - idle, 1),
                    "memory_used_gb": round(used_bytes / 1024**3, 1),
                    "memory_total_gb": round(total / 1024**3, 1),
                    "memory_pct": round(used_bytes / total * 100, 1),
                }
        except (ValueError, IndexError):
            pass
        return {}

    else:  # linux
        cmd = ("echo CPU=$(vmstat 1 2 | tail -1 | awk '{print 100-$15}'); "
               "free -b | awk '/Mem:/{print \"MEMTOTAL=\"$2; print \"MEMAVAIL=\"$7}'")
        result = executor.run(cmd, timeout=12)
        kv = _parse_kv_lines(result.stdout)
        try:
            total = float(kv.get("MEMTOTAL", 0))
            avail = float(kv.get("MEMAVAIL", 0))
            if total > 0:
                return {
                    "cpu_pct": float(kv.get("CPU", 0)),
                    "memory_used_gb": round((total - avail) / 1024 / 1024 / 1024, 1),
                    "memory_total_gb": round(total / 1024 / 1024 / 1024, 1),
                    "memory_pct": round((total - avail) / total * 100, 1),
                }
        except ValueError:
            pass
        return {}


# ==================== 推理指标 ====================

def _collect_vllm_metrics(m: dict) -> dict:
    """vLLM 引擎指标映射（Prometheus 格式，vllm: 前缀）

    vLLM 暴露的累计 Counter 与 histogram _sum/_count，换算出与 llama.cpp
    同构的字段，供前端复用同一套曲线。vLLM 无投机解码/前缀缓存命中率概念，
    对应字段返回 0（前端据此不绘制）。
    """
    prompt_tokens = m.get("vllm:prompt_tokens_total", 0)
    completion_tokens = m.get("vllm:generation_tokens_total", 0)
    # 耗时直方图的累计和（秒）
    prompt_seconds = m.get("vllm:time_to_first_token_seconds_sum", 0)
    e2e_seconds = m.get("vllm:e2e_request_latency_seconds_sum", 0)

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "prompt_speed": round(prompt_tokens / prompt_seconds, 2) if prompt_seconds > 0 else 0,
        "completion_speed": round(completion_tokens / e2e_seconds, 2) if e2e_seconds > 0 else 0,
        "cache_hit_rate": 0,   # vLLM 暂不暴露等价指标
        "spec_accept_rate": 0,  # vLLM 投机解码指标口径不同，暂不采集
    }


def _collect_sglang_metrics(m: dict) -> dict:
    """SGLang 引擎指标映射（Prometheus 格式，sglang: 前缀）

    依据官方文档 references/production_metrics：SGLang 以 --enable-metrics 启动后
    暴露 sglang:prompt_tokens_total / generation_tokens_total（counter）、
    time_to_first_token_seconds / e2e_request_latency_seconds（histogram，
    取 _sum）、cache_hit_rate（gauge，取值 0~1）。

    与 vLLM 分支的差异：SGLang 直接给出 RadixAttention 前缀缓存命中率，
    因此 cache_hit_rate 有真实值（换算成与 llama.cpp 一致的百分比口径）；
    投机解码口径不同，返回 0。速度口径与 vLLM 分支保持一致。
    """
    prompt_tokens = m.get("sglang:prompt_tokens_total", 0)
    completion_tokens = m.get("sglang:generation_tokens_total", 0)
    # 耗时直方图的累计和（秒）
    prompt_seconds = m.get("sglang:time_to_first_token_seconds_sum", 0)
    e2e_seconds = m.get("sglang:e2e_request_latency_seconds_sum", 0)
    # 官方文档示例为 0~1 的比值，前端与 llama.cpp 一致按百分比展示
    cache_hit = m.get("sglang:cache_hit_rate", 0)

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "prompt_speed": round(prompt_tokens / prompt_seconds, 2) if prompt_seconds > 0 else 0,
        "completion_speed": round(completion_tokens / e2e_seconds, 2) if e2e_seconds > 0 else 0,
        "cache_hit_rate": round(cache_hit * 100, 1) if cache_hit else 0,
        "spec_accept_rate": 0,  # SGLang 投机解码指标口径不同，暂不采集
    }


def _metrics_url(target: Target) -> str:
    return f"http://127.0.0.1:{target.service_port}/metrics"


def _comfy_base(target: Target) -> str:
    port = target.service_port or 8188
    return f"http://127.0.0.1:{port}"


def _comfy_curl_json(executor: Executor, target: Target, path: str, timeout: int = 8):
    """在目标机上 curl ComfyUI 的某个 JSON 端点；失败返回 None。"""
    url = _comfy_base(target) + path
    cmd = f'curl -s --max-time {timeout} "{url}"'
    result = executor.run(cmd, timeout=timeout + 4)
    out = (result.stdout or "").strip()
    if not out or not out.startswith("{"):
        return None
    try:
        return json.loads(out)
    except (ValueError, json.JSONDecodeError):
        return None


def _collect_comfyui_metrics(executor: Executor, target: Target) -> dict:
    """ComfyUI 引擎级概览：在线状态 + 队列长度 + 显存占用。

    ComfyUI 没有 token/缓存/投机采样概念，返回的字段与文本引擎不同构，
    前端据 engine 字段切换到视频概览卡片。单个生成任务的逐步进度由
    /api/deploy/generate/progress 接口提供（基于 prompt_id 轮询）。"""
    stats = _comfy_curl_json(executor, target, "/system_stats")
    if stats is None:
        # 服务未响应
        return {"engine": "comfyui", "online": False}

    # /system_stats.devices[] 含 vram_total / vram_free（字节）
    vram_used_gb = vram_total_gb = 0.0
    devices = stats.get("devices") or []
    for dev in devices:
        vt = float(dev.get("vram_total", 0) or 0)
        vf = float(dev.get("vram_free", 0) or 0)
        if vt > 0:
            vram_total_gb = round(vt / 1024**3, 1)
            vram_used_gb = round((vt - vf) / 1024**3, 1)
            break

    running = pending = 0
    q = _comfy_curl_json(executor, target, "/queue")
    if q:
        running = len(q.get("queue_running") or [])
        pending = len(q.get("queue_pending") or [])

    return {
        "engine": "comfyui",
        "online": True,
        "queue_running": running,
        "queue_pending": pending,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
    }


def collect_metrics(executor: Executor, target: Target) -> dict:
    """在目标机器上 curl metrics（metrics 端口仅目标机本地可访问）"""
    # ComfyUI 是 JSON API，非 Prometheus 文本，单独分流
    if getattr(target, "engine_type", "") == "comfyui":
        return _collect_comfyui_metrics(executor, target)

    cmd = f'curl -s --max-time 5 {_metrics_url(target)}'
    result = executor.run(cmd, timeout=8)
    if not result.ok or not result.stdout:
        return {}

    m = {}
    for line in result.stdout.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        parts = line.split(" ")
        if len(parts) >= 2:
            try:
                val = float(parts[1])
            except ValueError:
                continue
            # Prometheus 指标可能带标签（vLLM / SGLang 均输出 {model_name="..."}）。
            # 统一按去掉标签后的指标名归并，同名多标签累加（单模型部署下等价于取值），
            # 否则 m 里存的是「指标名{标签}」，按名取值会永远拿到 0。
            name = parts[0].split("{", 1)[0]
            m[name] = m.get(name, 0.0) + val

    # 引擎感知：vLLM / SGLang 与 llama.cpp 的指标前缀与名称不同
    engine_type = getattr(target, "engine_type", "llama_cpp") or "llama_cpp"
    if engine_type == "vllm":
        return _collect_vllm_metrics(m)
    if engine_type == "sglang":
        return _collect_sglang_metrics(m)

    prompt_tokens = m.get("llamacpp:prompt_tokens_total", 0)
    completion_tokens = m.get("llamacpp:tokens_predicted_total", 0)
    prompt_seconds = m.get("llamacpp:prompt_seconds_total", 0)
    predict_seconds = m.get("llamacpp:tokens_predicted_seconds_total", 0)
    cached_tokens = m.get("llamacpp:prompt_tokens_cached_total", 0)
    spec_draft = m.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
    spec_accepted = m.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)

    total_prompt = prompt_tokens + cached_tokens
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "prompt_speed": round(prompt_tokens / prompt_seconds, 2) if prompt_seconds > 0 else 0,
        "completion_speed": round(completion_tokens / predict_seconds, 2) if predict_seconds > 0 else 0,
        "cache_hit_rate": round(cached_tokens / total_prompt * 100, 1) if total_prompt > 0 else 0,
        "spec_accept_rate": round(spec_accepted / spec_draft * 100, 1) if spec_draft > 0 else 0,
    }


# ==================== 硬件检测（首页用） ====================

def _detect_cpu_static(executor: Executor, target: Target) -> dict:
    if target.os == "windows":
        result = executor.run(
            _ps("$p=Get-CimInstance Win32_Processor; "
                "Write-Output ('Name=' + $p.Name); "
                "Write-Output ('NumberOfCores=' + $p.NumberOfCores); "
                "Write-Output ('NumberOfLogicalProcessors=' + $p.NumberOfLogicalProcessors)"),
            timeout=20)
        kv = _parse_kv_lines(result.stdout)
        return {
            "name": kv.get("Name", ""),
            "cores": int(kv.get("NumberOfCores", 0) or 0),
            "threads": int(kv.get("NumberOfLogicalProcessors", 0) or 0),
        }
    elif target.os == "macos":
        cmd = ("echo NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || "
               "sysctl -n hw.model); "
               "echo CORES=$(sysctl -n hw.physicalcpu); "
               "echo THREADS=$(sysctl -n hw.logicalcpu)")
        result = executor.run(cmd, timeout=10)
        kv = _parse_kv_lines(result.stdout)
        return {
            "name": kv.get("NAME", ""),
            "cores": int(kv.get("CORES", 0) or 0),
            "threads": int(kv.get("THREADS", 0) or 0),
        }
    else:  # linux
        cmd = ("echo NAME=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'); "
               "echo CORES=$(grep -c ^processor /proc/cpuinfo)")
        result = executor.run(cmd, timeout=10)
        kv = _parse_kv_lines(result.stdout)
        cores = int(kv.get("CORES", 0) or 0)
        return {"name": kv.get("NAME", ""), "cores": cores, "threads": cores}


def _detect_memory_static(executor: Executor, target: Target) -> dict:
    if target.os == "windows":
        result = executor.run(
            _ps("$o=Get-CimInstance Win32_OperatingSystem; "
                "Write-Output ('TotalVisibleMemorySize=' + $o.TotalVisibleMemorySize); "
                "Write-Output ('FreePhysicalMemory=' + $o.FreePhysicalMemory)"),
            timeout=20)
        kv = _parse_kv_lines(result.stdout)
        total = float(kv.get("TotalVisibleMemorySize", 0) or 0)
        free = float(kv.get("FreePhysicalMemory", 0) or 0)
        return {"total_gb": round(total / 1024 / 1024, 1), "free_gb": round(free / 1024 / 1024, 1)}
    elif target.os == "macos":
        result = executor.run("echo TOTAL=$(sysctl -n hw.memsize)", timeout=10)
        kv = _parse_kv_lines(result.stdout)
        total = float(kv.get("TOTAL", 0) or 0)
        return {"total_gb": round(total / 1024**3, 1), "free_gb": 0}
    else:  # linux
        result = executor.run("free -b | awk '/Mem:/{print \"TOTAL=\"$2; print \"AVAIL=\"$7}'", timeout=10)
        kv = _parse_kv_lines(result.stdout)
        total = float(kv.get("TOTAL", 0) or 0)
        avail = float(kv.get("AVAIL", 0) or 0)
        return {"total_gb": round(total / 1024**3, 1), "free_gb": round(avail / 1024**3, 1)}


def _detect_gpu_windows(executor: Executor, target: Target) -> dict:
    """Windows 上非 NVIDIA 显卡（AMD/Intel）的静态识别。

    nvidia-smi 只覆盖 NVIDIA，其余厂商没有等价的命令行工具，改用 WMI 列显示
    适配器取名称与驱动版本。显存优先读注册表 HardwareInformation.qwMemorySize
    （64 位），因为 Win32_VideoController.AdapterRAM 是 uint32，显存超过 4GB 会
    回绕成 0 或小值——16GB 级别的卡正好踩中这个坑。剩余显存拿不到，留 0。
    多适配器（如集显 + 独显）时取显存最大的一块，与「跑模型的那块」一致。
    """
    result = executor.run(_ps(
        "$ad=Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -notmatch 'Basic Display|Remote Display|Virtual' }; "
        f"$m=(Get-ChildItem '{_WIN_DISPLAY_CLASS}' | "
        "ForEach-Object { (Get-ItemProperty $_.PSPath -Name "
        "'HardwareInformation.qwMemorySize' -ErrorAction SilentlyContinue)"
        ".'HardwareInformation.qwMemorySize' } | Measure-Object -Maximum).Maximum; "
        "$i=0; "
        "foreach ($v in $ad) { "
        "Write-Output ('GPU' + $i + '_NAME=' + $v.Name); "
        "Write-Output ('GPU' + $i + '_VENDOR=' + $v.AdapterCompatibility); "
        "Write-Output ('GPU' + $i + '_DRIVER=' + $v.DriverVersion); "
        "Write-Output ('GPU' + $i + '_MEM=' + $m); "
        "$i++ }"
    ), timeout=20)
    kv = _parse_kv_lines(result.stdout)
    best = None
    for i in range(10):  # 适配器数量有上限，逐个比对取显存最大的一块
        name = kv.get(f"GPU{i}_NAME", "").strip()
        if not name:
            continue
        mem_gb = _parse_mem_bytes(kv.get(f"GPU{i}_MEM", ""))
        if best is None or mem_gb > best["total_memory_gb"]:
            best = {
                "name": name,
                "total_memory_gb": mem_gb,
                "free_memory_gb": 0,
                "driver": (kv.get(f"GPU{i}_DRIVER", "") or "").strip(),
                "vendor": _normalize_gpu_vendor(kv.get(f"GPU{i}_VENDOR", "")),
            }
    return best


def _detect_gpu_static(executor: Executor, target: Target) -> dict:
    # NVIDIA 跨平台
    result = executor.run(
        "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version "
        "--format=csv,noheader,nounits", timeout=8)
    if result.ok and result.stdout:
        parts = [p.strip() for p in result.stdout.split(",")]
        if len(parts) >= 4:
            try:
                return {
                    "name": parts[0],
                    "total_memory_gb": round(float(parts[1]) / 1024, 1),
                    "free_memory_gb": round(float(parts[2]) / 1024, 1),
                    "driver": parts[3],
                }
            except ValueError:
                pass

    # Windows 上的 AMD/Intel 卡：nvidia-smi 不存在，改用显示适配器列表
    if target.os == "windows":
        gpu = _detect_gpu_windows(executor, target)
        if gpu:
            return gpu

    # macOS Apple Silicon：统一内存，无独立显存，返回芯片 GPU 信息
    if target.os == "macos":
        result = executor.run(
            "system_profiler SPDisplaysDataType 2>/dev/null | "
            "awk '/Chipset Model/{gsub(/^[ \t]+Chipset Model: /,\"\"); name=name\" \"$0} "
            "/Vendor/{next} END{print \"NAME=\"name}'", timeout=12)
        kv = _parse_kv_lines(result.stdout)
        name = kv.get("NAME", "").strip()
        if name:
            return {"name": name, "total_memory_gb": 0, "free_memory_gb": 0,
                    "driver": "Apple", "unified": True}
    return None


def _detect_disk(executor: Executor, target: Target) -> dict:
    if target.os == "windows":
        drive = target.models_dir[:2] if len(target.models_dir) >= 2 else "C:"
        result = executor.run(
            _ps(f"$d=Get-CimInstance Win32_LogicalDisk | Where-Object DeviceID -eq '{drive}'; "
                "Write-Output ('Size=' + $d.Size); "
                "Write-Output ('FreeSpace=' + $d.FreeSpace)"),
            timeout=20)
        kv = _parse_kv_lines(result.stdout)
        try:
            return {
                "total_gb": round(float(kv.get("Size", 0)) / 1024**3, 1),
                "free_gb": round(float(kv.get("FreeSpace", 0)) / 1024**3, 1),
            }
        except ValueError:
            return {}
    else:  # macOS / Linux 都用 df
        path = target.models_dir or "/"
        result = executor.run(
            f'df -k "{path}" 2>/dev/null | tail -1 | '
            'awk \'{print "TOTAL="$2; print "AVAIL="$4}\'', timeout=8)
        kv = _parse_kv_lines(result.stdout)
        try:
            return {
                "total_gb": round(float(kv.get("TOTAL", 0)) * 1024 / 1024**3, 1),
                "free_gb": round(float(kv.get("AVAIL", 0)) * 1024 / 1024**3, 1),
            }
        except ValueError:
            return {}


def detect_hardware(executor: Executor, target: Target) -> dict:
    """完整硬件检测"""
    info = {}
    info["gpu"] = _detect_gpu_static(executor, target)
    info["cpu"] = _detect_cpu_static(executor, target)
    info["memory"] = _detect_memory_static(executor, target)
    info["disk"] = _detect_disk(executor, target)
    return info


def collect_all(executor: Executor, target: Target) -> dict:
    """监控快照"""
    return {
        "gpu": _collect_gpu(executor, target),
        "cpu_mem": _collect_cpu_mem(executor, target),
        "metrics": collect_metrics(executor, target),
    }
