"""推理引擎检测与一键安装

面向所有用户：若目标机尚未安装 llama.cpp（llama-server），
提供一键安装。按目标 OS 选择安装方式：
  - Windows：下载官方预编译 CUDA 包并解压
  - macOS：Homebrew 安装
  - Linux：源码编译

安装为耗时操作，采用后台线程执行 + 日志轮询，避免 HTTP 超时。
"""

import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional, List

from .executor import Executor
from .collectors import detect_gpu_vendor
from ..models.target import Target
from .i18n import L, set_lang

# 全局安装任务表：job_id -> {status, logs, target_id, result}
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


# ==================== 检测 ====================

def detect_engine(executor: Executor, target: Target) -> dict:
    """检测目标机是否已安装所选推理引擎（按 engine_type 分发）"""
    engine_type = getattr(target, "engine_type", "llama_cpp") or "llama_cpp"
    if engine_type == "vllm":
        return _detect_vllm(executor, target)
    if engine_type == "sglang":
        return _detect_sglang(executor, target)
    if engine_type == "comfyui":
        return _detect_comfyui(executor, target)
    return _detect_llama(executor, target)


def _detect_comfyui(executor: Executor, target: Target) -> dict:
    """检测 ComfyUI 是否已安装：安装根目录下的 main.py 入口是否存在。
    engine_path 对 ComfyUI 语义是安装根目录（非可执行文件）。"""
    d = target.engine_path
    if not d:
        return {"installed": False, "engine": "comfyui", "path": "", "version": "",
                "reason": L("detect.no_comfyui_dir")}
    main_py = _join(d, "main.py")
    if target.os == "windows":
        found = "FOUND" in executor.run(f'if exist "{main_py}" (echo FOUND)').stdout
    else:
        found = "FOUND" in executor.run(f'test -f "{main_py}" && echo FOUND').stdout
    version = ""
    if found:
        # 读 ComfyUI 版本号文件（commit 短哈希）
        vr = executor.run(f'cd "{d}" && git rev-parse --short HEAD 2>&1', timeout=10)
        if vr.ok:
            version = vr.stdout.strip()
    return {
        "installed": found, "engine": "comfyui", "path": d, "version": version,
        "reason": "" if found else L("detect.comfyui_missing"),
    }


def _join(base: str, name: str) -> str:
    """跨平台路径拼接（Windows 反斜杠 / 其他正斜杠）。"""
    if "\\" in base or ":" in base and "/" not in base:
        sep = "\\"
    elif base.endswith("/") or base.endswith("\\"):
        sep = ""
    else:
        sep = "/"
    if sep == "":
        return base + name
    return base.rstrip("/\\") + sep + name


def _detect_llama(executor: Executor, target: Target) -> dict:
    """检测 llama-server 二进制是否存在"""
    exe = target.engine_path
    if not exe:
        return {"installed": False, "engine": "llama_cpp", "reason": L("detect.no_engine_path"), "path": ""}

    if target.os == "windows":
        result = executor.run(f'if exist "{exe}" (echo FOUND)')
        found = "FOUND" in result.stdout
    else:
        result = executor.run(f'test -f "{exe}" && echo FOUND')
        found = "FOUND" in result.stdout

    version = ""
    if found:
        vr = executor.run(f'"{exe}" --version 2>&1 | head -1', timeout=10)
        version = vr.stdout.strip()

    return {
        "installed": found,
        "engine": "llama_cpp",
        "path": exe,
        "version": version,
        "reason": "" if found else L("detect.llama_missing"),
    }


def _detect_vllm(executor: Executor, target: Target) -> dict:
    """检测 vLLM 是否可用（pip 安装后 vllm 命令在 PATH）"""
    cmd = target.engine_path or "vllm"
    if target.os == "windows":
        # vLLM 不支持 Windows 原生，提示走 WSL2
        return {
            "installed": False, "engine": "vllm", "path": cmd, "version": "",
            "reason": L("detect.vllm_windows"),
            "windows_note": True,
        }
    result = executor.run(f"{cmd} --version 2>&1", timeout=25)
    out = (result.stdout or "").lower()
    not_found = "not found" in out or "no module" in out or "command not found" in out
    installed = result.ok and any(c.isdigit() for c in out) and not not_found
    version = ""
    if installed:
        for ln in result.stdout.splitlines():
            if any(c.isdigit() for c in ln):
                version = ln.strip()
                break
    return {
        "installed": installed, "engine": "vllm", "path": cmd, "version": version,
        "reason": "" if installed else L("detect.vllm_missing"),
    }


def _detect_sglang(executor: Executor, target: Target) -> dict:
    """检测 SGLang 是否可用（pip/uv 安装后 sglang 命令在 PATH）"""
    cmd = target.engine_path or "sglang"
    if target.os == "windows":
        # SGLang 官方安装说明面向 Linux + NVIDIA GPU
        return {
            "installed": False, "engine": "sglang", "path": cmd, "version": "",
            "reason": L("detect.sglang_windows"),
            "windows_note": True,
        }
    result = executor.run(f"{cmd} --version 2>&1", timeout=25)
    out = (result.stdout or "").lower()
    not_found = "not found" in out or "no module" in out or "command not found" in out
    installed = result.ok and any(c.isdigit() for c in out) and not not_found
    version = ""
    if installed:
        for ln in result.stdout.splitlines():
            if any(c.isdigit() for c in ln):
                version = ln.strip()
                break
    return {
        "installed": installed, "engine": "sglang", "path": cmd, "version": version,
        "reason": "" if installed else L("detect.sglang_missing"),
    }


# ==================== 安装任务管理 ====================

def _append_log(job_id: str, line: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": line})


def _run_step(executor: Executor, job_id: str, cmd: str, desc: str,
              timeout: int = 600, check: bool = False):
    """执行一步并记录日志，返回 ExecResult

    check=True：该步失败立刻抛错并带上真实错误。用于「失败后继续跑也没意义」
    的步骤（建目录、下载、解压、编译），否则错误会被拖到最后一步，
    报成「找不到 llama-server」这种看不出原因的消息。
    """
    _append_log(job_id, f"▶ {desc}")
    result = executor.run(cmd, timeout=timeout)
    for ln in (result.stdout or "").splitlines()[-5:]:
        if ln.strip():
            _append_log(job_id, f"  {ln.strip()}")
    if not result.ok:
        errs = [ln.strip() for ln in (result.stderr or "").splitlines()[-5:] if ln.strip()]
        for ln in errs:
            _append_log(job_id, f"  [err] {ln}")
        if check:
            tail = (" | ".join(errs[-2:]) or (result.stdout or "").strip()[-200:]
                    or f"exit={result.returncode}")
            raise RuntimeError(L("install.err.step_failed", desc=desc, err=tail))
    return result


def get_job(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> List[dict]:
    with _LOCK:
        return [{"job_id": j["job_id"], "status": j["status"],
                 "target_id": j["target_id"]} for j in _JOBS.values()]


# ==================== 后端选择与下载源 ====================

# llama.cpp 的 GPU 后端；auto 交给 resolve_backend 推断
BACKENDS = ("auto", "cuda", "rocm", "vulkan", "cpu", "metal")

# 官方 release 里各平台+后端的 asset 名特征。
# 只匹配固定片段、不锚定小版本号：官方随 CUDA/ROCm 版本改过名
# （cu12 -> cuda-12.4 -> cuda-13.4），锚死会在上游发版后失配。
# 结尾的 -x64 同时排除 arm64 包。
_ASSET_PATTERNS = {
    ("windows", "cuda"):   r"^llama-.*-bin-win-cuda-\d+\.\d+-x64\.zip$",
    ("windows", "rocm"):   r"^llama-.*-bin-win-rocm-[\d.]+-x64\.zip$",
    ("windows", "vulkan"): r"^llama-.*-bin-win-vulkan-x64\.zip$",
    ("windows", "cpu"):    r"^llama-.*-bin-win-cpu-x64\.zip$",
    ("linux", "cuda"):     r"^llama-.*-bin-ubuntu-cuda-\d+\.\d+-x64\.tar\.gz$",
    ("linux", "rocm"):     r"^llama-.*-bin-ubuntu-rocm-[\d.]+-x64\.tar\.gz$",
    ("linux", "vulkan"):   r"^llama-.*-bin-ubuntu-vulkan-x64\.tar\.gz$",
    ("linux", "cpu"):      r"^llama-.*-bin-ubuntu-x64\.tar\.gz$",
}

# CUDA 构建要额外下运行时包（官方把 cudart 与主包分开发布）
# 注意命名不统一：Windows 是 cudart-llama-bin-win-...，Linux 带版本号
# cudart-llama-b11344-bin-ubuntu-...，所以中间用 .*bin- 兜住两种
_CUDART_PATTERNS = {
    "windows": r"^cudart-llama-.*bin-win-cuda-\d+\.\d+-x64\.zip$",
    "linux":   r"^cudart-llama-.*bin-ubuntu-cuda-\d+\.\d+-x64\.tar\.gz$",
}

_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page={n}"
_HTTP_UA = "ReadyLLM-installer"


def resolve_backend(target: Target, gpu_vendor: str = "") -> str:
    """解析该目标机该用哪个 llama.cpp 构建

    显式配置优先；auto 时按系统与显卡厂商推断：macOS -> metal，
    NVIDIA -> cuda，AMD / Intel -> vulkan（Vulkan 随显卡驱动自带，
    不必先装 ROCm 运行时，对普通用户门槛最低），识别不到 -> cpu。
    装了 ROCm 的 A 卡用户可在设置里显式选 rocm。
    """
    explicit = (getattr(target, "llama_backend", "") or "auto").strip().lower()
    if explicit in BACKENDS and explicit != "auto":
        return explicit
    if target.os == "macos":
        return "metal"
    v = (gpu_vendor or "").strip().lower()
    if v == "nvidia":
        return "cuda"
    if v in ("amd", "intel"):
        return "vulkan"
    return "cpu"


def pick_assets(releases: list, os_name: str, backend: str) -> Optional[dict]:
    """从 releases 列表里挑出该平台+后端要下载的包（纯函数，便于单测）

    返回 {"tag", "primary": {name,url}, "cudart": {name,url}|None}；挑不到返回 None。

    必须从新到旧遍历，不能只看 releases/latest：官方会发不带平台二进制的
    版本（例如只挂 UI 包的 tag），只看 latest 会直接扑空。
    """
    pat = _ASSET_PATTERNS.get((os_name, backend))
    if not pat:
        return None
    rx = re.compile(pat)
    cudart_rx = re.compile(_CUDART_PATTERNS.get(os_name, "$^"))  # 无该平台时永不匹配
    for rel in releases or []:
        primary = None
        cudart = None
        for a in rel.get("assets") or []:
            name = a.get("name") or ""
            url = a.get("browser_download_url") or ""
            if not url:
                continue
            if primary is None and rx.match(name):
                primary = {"name": name, "url": url}
            elif cudart is None and backend == "cuda" and cudart_rx.match(name):
                cudart = {"name": name, "url": url}
        if primary:
            return {"tag": rel.get("tag_name") or "", "primary": primary, "cudart": cudart}
    return None


def fetch_releases(limit: int = 20, timeout: int = 25) -> list:
    """拉取 llama.cpp 官方 release 列表（控制端出网）

    失败抛 RuntimeError，由调用方决定是否回退到目标机自查。
    """
    req = urllib.request.Request(
        _RELEASES_API.format(n=limit),
        headers={"User-Agent": _HTTP_UA, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GitHub API HTTP {e.code}")
    except Exception as e:
        raise RuntimeError(f"GitHub API unreachable: {e}")


# ==================== 各平台安装脚本 ====================

def _pick_release(job_id: str, os_name: str, backend: str) -> dict:
    """查官方 release 列表，挑出该平台+后端要下的包

    从新到旧遍历，不能只看 releases/latest：官方会发只挂 UI 包的版本，
    latest 上根本没有平台二进制（实测 v0.5.0 只有 1 个 asset）。
    """
    _append_log(job_id, "▶ " + L("install.query_release"))
    try:
        releases = fetch_releases()
    except RuntimeError as e:
        raise RuntimeError(L("install.err.release_api", err=e))
    picked = pick_assets(releases, os_name, backend)
    if not picked:
        raise RuntimeError(L("install.err.no_download_url", backend=backend))
    _append_log(job_id, "  " + L("install.release_tag", tag=picked["tag"]))
    return picked


def _win_download_cmd(url: str, dest: str) -> str:
    """Windows 下载命令

    显式打开 TLS 1.2（PowerShell 5.1 默认不开，Windows 上不写这段会直接失败），
    并带 User-Agent——GitHub 的下载 CDN 对无 UA 的请求会返回 403。
    """
    return (
        'powershell -Command "'
        "$ProgressPreference='SilentlyContinue'; "
        "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; "
        f"Invoke-WebRequest -Uri '{url}' -OutFile '{dest}' -UseBasicParsing "
        f"-Headers @{{'User-Agent'='{_HTTP_UA}'}}\""
    )


def _win_find_server(executor: Executor, install_dir: str) -> str:
    """在解压结果里定位 llama-server.exe（官方包解压后带一层版本子目录）"""
    fr = executor.run(
        f'powershell -Command "Get-ChildItem -Path {install_dir} -Recurse -Filter llama-server.exe '
        '| Select-Object -First 1 -ExpandProperty FullName"', timeout=30)
    for ln in fr.stdout.splitlines():
        if ln.strip().lower().endswith("llama-server.exe"):
            return ln.strip()
    return ""


def _install_windows(executor: Executor, job_id: str, target: Target) -> str:
    """下载官方预编译包并解压，返回安装后的 engine_path

    按解析出的后端挑包（cuda / rocm / vulkan / cpu），不再固定 CUDA。
    """
    backend = resolve_backend(target, detect_gpu_vendor(executor, target))
    _append_log(job_id, "▶ " + L("install.backend_selected", backend=backend))

    install_dir = r"C:\llama"
    _run_step(executor, job_id,
              f'powershell -Command "New-Item -ItemType Directory -Force -Path {install_dir} | Out-Null"',
              L("install.mkdir", dir=install_dir), check=True)

    picked = _pick_release(job_id, "windows", backend)
    _append_log(job_id, "  " + L("install.package_name", name=picked["primary"]["name"]))

    zip_path = _join(install_dir, "llama.zip")
    _run_step(executor, job_id, _win_download_cmd(picked["primary"]["url"], zip_path),
              L("install.download_pkg"), timeout=900, check=True)

    _run_step(executor, job_id,
              f'powershell -Command "Expand-Archive -Path \'{zip_path}\' -DestinationPath \'{install_dir}\' -Force"',
              L("install.unzip"), check=True)

    exe_path = _win_find_server(executor, install_dir)
    if not exe_path:
        raise RuntimeError(L("install.err.no_llama_exe"))

    # CUDA 包必须额外铺一层运行时 dll（官方把 cudart 与主包分开发布），
    # 铺到 exe 同目录，否则 llama-server.exe 一启动就缺 dll
    if picked.get("cudart"):
        exe_dir = exe_path.rsplit("\\", 1)[0]
        cudart_zip = _join(install_dir, "cudart.zip")
        _run_step(executor, job_id, _win_download_cmd(picked["cudart"]["url"], cudart_zip),
                  L("install.download_cudart"), timeout=900, check=True)
        _run_step(executor, job_id,
                  f'powershell -Command "Expand-Archive -Path \'{cudart_zip}\' -DestinationPath \'{exe_dir}\' -Force"',
                  L("install.unzip_cudart"), check=True)

    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


def _install_sglang(executor: Executor, job_id: str, target: Target) -> str:
    """pip 安装 SGLang（仅 Linux/macOS，Windows 不支持原生运行）

    官方文档要求 Python 3.10+ 与 CUDA 环境，安装方式为 pip/uv
    （uv pip install --prerelease=allow sglang，pip 等价）。
    """
    if target.os == "windows":
        raise RuntimeError(L("install.err.sglang_windows"))

    _run_step(executor, job_id, "command -v pip3 || command -v pip",
              L("install.check_pip"), timeout=20)

    # 检测 CUDA / NVIDIA GPU（SGLang 主要面向 NVIDIA）
    cuda = executor.run(
        "command -v nvcc && nvidia-smi --query-gpu=name --format=csv,noheader", timeout=15)
    if cuda.stdout.strip():
        _append_log(job_id, "  " + L("install.gpu_detected", info=cuda.stdout.strip().splitlines()[0]))
    else:
        _append_log(job_id, "  " + L("install.no_cuda_sglang"))

    _run_step(executor, job_id,
              "pip3 install -U sglang 2>&1 | tail -20 || pip install -U sglang 2>&1 | tail -20",
              L("install.pip_sglang"), timeout=3600)

    fr = executor.run("command -v sglang")
    exe_path = fr.stdout.strip().splitlines()[-1] if fr.stdout.strip() else ""
    if not exe_path:
        raise RuntimeError(L("install.err.no_sglang_cmd"))
    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


def _install_macos(executor: Executor, job_id: str, target: Target) -> str:
    """Homebrew 安装 llama.cpp"""
    # 检查 brew
    brew_check = executor.run("command -v brew")
    if not brew_check.stdout:
        raise RuntimeError(L("install.err.no_brew"))

    _run_step(executor, job_id, "brew install llama.cpp", L("install.brew_llama"), timeout=1200)

    # 定位可执行文件
    fr = executor.run("command -v llama-server")
    exe_path = fr.stdout.strip().splitlines()[-1] if fr.stdout.strip() else ""
    if not exe_path:
        raise RuntimeError(L("install.err.no_llama_brew"))
    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


def _nix_download_cmd(url: str, path: str) -> str:
    """类 Unix 下载命令（curl 优先、wget 兜底，都带失败即退出）"""
    return (f"curl -fL --retry 3 --retry-delay 2 -o '{path}' '{url}' "
            f"|| wget -O '{path}' '{url}'")


def _nix_find_server(executor: Executor, root: str) -> str:
    """在目录内定位 llama-server 文件（找到就补可执行位）"""
    fr = executor.run(f'find {root} -name llama-server -type f 2>/dev/null | head -1')
    exe = fr.stdout.strip().splitlines()[-1] if fr.stdout.strip() else ""
    if exe:
        executor.run(f'chmod +x "{exe}"')
    return exe


def _compile_llama_linux(executor: Executor, job_id: str, install_dir: str,
                         backend: str) -> str:
    """源码编译 llama.cpp（下不到预编译包或包内缺 llama-server 时的兜底）

    编译开关按后端给：cuda / rocm / vulkan；cpu 或未知则纯 CPU 构建。
    """
    _run_step(executor, job_id, "command -v cmake && command -v git && command -v g++",
              L("install.check_build_deps"), timeout=30, check=True)

    src = f"{install_dir}/src"
    _run_step(executor, job_id,
              f"rm -rf {src} && git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}",
              L("install.clone_llama"), timeout=600, check=True)

    flags = {
        "cuda": "-DGGML_CUDA=ON",
        "rocm": "-DGGML_HIP=ON",      # 需 ROCm 工具链；AMDGPU_TARGETS 交给 cmake 探测
        "vulkan": "-DGGML_VULKAN=ON",  # 需 vulkan 开发包
    }.get(backend, "")
    _run_step(executor, job_id,
              f"cd {src} && cmake -B build {flags} && "
              f"cmake --build build --config Release -j --target llama-server",
              L("install.compile_llama_gpu", backend=backend) if flags else L("install.compile_llama"),
              timeout=3600, check=True)

    exe_path = f"{src}/build/bin/llama-server"
    check = executor.run(f'test -f "{exe_path}" && echo FOUND')
    if "FOUND" not in check.stdout:
        raise RuntimeError(L("install.err.no_compile_out"))
    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


def _install_linux(executor: Executor, job_id: str, target: Target) -> str:
    """安装 llama.cpp（Linux）

    优先下官方预编译包（秒级，官方对 ubuntu 有 cuda / rocm / vulkan / cpu 四种构建），
    下不到、架构不匹配或包内缺 llama-server 时回退源码编译。
    """
    backend = resolve_backend(target, detect_gpu_vendor(executor, target))
    _append_log(job_id, "▶ " + L("install.backend_selected", backend=backend))

    install_dir = "$HOME/.local/share/readyllm/llama"
    _run_step(executor, job_id, f"mkdir -p {install_dir}",
              L("install.mkdir", dir=install_dir), timeout=30, check=True)

    picked = None
    arch = (executor.run("uname -m").stdout or "").strip()
    if "x86_64" in arch:
        try:
            picked = _pick_release(job_id, "linux", backend)
        except RuntimeError as e:
            _append_log(job_id, "  " + L("install.prebuilt_failed", err=e))
    else:
        _append_log(job_id, "  " + L("install.arch_no_prebuilt", arch=arch or "unknown"))

    if picked:
        tgz = f"{install_dir}/llama.tar.gz"
        _append_log(job_id, "  " + L("install.package_name", name=picked["primary"]["name"]))
        _run_step(executor, job_id, _nix_download_cmd(picked["primary"]["url"], tgz),
                  L("install.download_pkg"), timeout=900, check=True)
        _run_step(executor, job_id, f"tar -xzf '{tgz}' -C {install_dir}",
                  L("install.unzip"), check=True)
        exe = _nix_find_server(executor, install_dir)
        if exe:
            _append_log(job_id, "  " + L("install.engine_path", path=exe))
            return exe
        _append_log(job_id, "  " + L("install.prebuilt_missing"))

    _append_log(job_id, "▶ " + L("install.compile_fallback"))
    return _compile_llama_linux(executor, job_id, install_dir, backend)


def _install_vllm(executor: Executor, job_id: str, target: Target) -> str:
    """pip 安装 vLLM（仅 Linux/macOS，Windows 不支持原生运行）"""
    if target.os == "windows":
        raise RuntimeError(L("install.err.vllm_windows"))

    _run_step(executor, job_id, "command -v pip3 || command -v pip",
              L("install.check_pip"), timeout=20)

    # 检测 CUDA / NVIDIA GPU（vLLM 主要面向 NVIDIA）
    cuda = executor.run(
        "command -v nvcc && nvidia-smi --query-gpu=name --format=csv,noheader", timeout=15)
    if cuda.stdout.strip():
        _append_log(job_id, "  " + L("install.gpu_detected", info=cuda.stdout.strip().splitlines()[0]))
    else:
        _append_log(job_id, "  " + L("install.no_cuda_vllm"))

    _run_step(executor, job_id,
              "pip3 install -U vllm 2>&1 | tail -20 || pip install -U vllm 2>&1 | tail -20",
              L("install.pip_vllm"), timeout=3600)

    fr = executor.run("command -v vllm")
    exe_path = fr.stdout.strip().splitlines()[-1] if fr.stdout.strip() else ""
    if not exe_path:
        raise RuntimeError(L("install.err.no_vllm_cmd"))
    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


def _install_comfyui(executor: Executor, job_id: str, target: Target) -> str:
    """git clone ComfyUI + pip 安装依赖。返回安装根目录（作为 engine_path 回填）。

    安装目录：优先用用户已配置的 engine_path 作为目标目录，否则用通用默认位置
    （Windows: C:\\ComfyUI，类 Unix: ~/ComfyUI）。不写死任何个人机器路径。"""
    if target.engine_path:
        install_dir = target.engine_path
    elif target.os == "windows":
        install_dir = r"C:\ComfyUI"
    else:
        install_dir = "$HOME/ComfyUI"

    repo = "https://github.com/comfyanonymous/ComfyUI.git"

    # 1) 检查 git / python / pip
    _run_step(executor, job_id,
              "git --version && (python --version || python3 --version)",
              L("install.check_git_python"), timeout=30)

    # 2) 克隆（若目录已存在则跳过克隆，仅更新）
    if target.os == "windows":
        exist = executor.run(f'if exist "{install_dir}\\main.py" (echo FOUND)').stdout
        if "FOUND" in exist:
            _append_log(job_id, "  " + L("install.comfyui_exists"))
        else:
            _run_step(executor, job_id,
                      f'git clone --depth 1 {repo} "{install_dir}"',
                      L("install.clone_comfyui"), timeout=900)
    else:
        exist = executor.run(f'test -f {install_dir}/main.py && echo FOUND').stdout
        if "FOUND" in exist:
            _append_log(job_id, "  " + L("install.comfyui_exists"))
        else:
            _run_step(executor, job_id,
                      f"git clone --depth 1 {repo} {install_dir}",
                      L("install.clone_comfyui"), timeout=900)

    # 3) pip 安装依赖（torch 等大依赖，耗时较长）
    py = "python" if target.os == "windows" else "python3"
    req = _join(install_dir, "requirements.txt") if target.os == "windows" else f"{install_dir.rstrip('/')}/requirements.txt"
    _run_step(executor, job_id,
              f'cd "{install_dir}" && {py} -m pip install -r "{req}" 2>&1 | tail -20'
              if target.os == "windows" else
              f"cd {install_dir} && {py} -m pip install -r requirements.txt 2>&1 | tail -20",
              L("install.pip_comfyui"), timeout=3600)

    # 4) 校验入口
    if target.os == "windows":
        chk = executor.run(f'if exist "{install_dir}\\main.py" (echo FOUND)')
    else:
        chk = executor.run(f'test -f {install_dir}/main.py && echo FOUND')
    if "FOUND" not in chk.stdout:
        raise RuntimeError(L("install.err.no_comfyui_main"))
    _append_log(job_id, "  " + L("install.comfyui_dir", path=install_dir))
    return install_dir


# ==================== 后台执行 ====================

def start_install(target: Target, lang: str = "zh") -> str:
    """启动安装任务，返回 job_id

    lang 是发起安装时的界面语言：后台线程不继承请求上下文，
    所以在 worker 开头显式设置，让整条安装日志按同一语言渲染。
    """
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "logs": [],
            "target_id": target.id,
            "engine_path": "",
            "error": "",
        }

    def _worker():
        executor = None
        set_lang(lang)
        try:
            from .executor import make_executor
            executor = make_executor(target)
            engine_type = getattr(target, "engine_type", "llama_cpp") or "llama_cpp"
            if engine_type == "vllm":
                _append_log(job_id, L("install.start", name=target.name, os=target.os, engine="vLLM"))
                exe = _install_vllm(executor, job_id, target)
            elif engine_type == "sglang":
                _append_log(job_id, L("install.start", name=target.name, os=target.os, engine="SGLang"))
                exe = _install_sglang(executor, job_id, target)
            elif engine_type == "comfyui":
                _append_log(job_id, L("install.start", name=target.name, os=target.os, engine="ComfyUI"))
                exe = _install_comfyui(executor, job_id, target)
            else:
                _append_log(job_id, L("install.start", name=target.name, os=target.os, engine="llama.cpp"))
                if target.os == "windows":
                    exe = _install_windows(executor, job_id, target)
                elif target.os == "macos":
                    exe = _install_macos(executor, job_id, target)
                else:
                    exe = _install_linux(executor, job_id, target)

            # 回填 engine_path 到配置
            target.engine_path = exe
            from ..models.target import upsert_target
            upsert_target(target)

            with _LOCK:
                job = _JOBS[job_id]
                job["status"] = "success"
                job["engine_path"] = exe
            _append_log(job_id, L("install.done"))
        except Exception as e:
            with _LOCK:
                job = _JOBS[job_id]
                job["status"] = "failed"
                job["error"] = str(e)
            _append_log(job_id, L("install.failed", err=e))
        finally:
            if executor:
                executor.close()

    threading.Thread(target=_worker, daemon=True).start()
    return job_id
