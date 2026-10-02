"""推理引擎检测与一键安装

面向所有用户：若目标机尚未安装 llama.cpp（llama-server），
提供一键安装。按目标 OS 选择安装方式：
  - Windows：下载官方预编译 CUDA 包并解压
  - macOS：Homebrew 安装
  - Linux：源码编译

安装为耗时操作，采用后台线程执行 + 日志轮询，避免 HTTP 超时。
"""

import threading
import time
import uuid
from typing import Optional, List

from .executor import Executor
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


def _run_step(executor: Executor, job_id: str, cmd: str, desc: str, timeout: int = 600):
    """执行一步并记录日志，返回 ExecResult"""
    _append_log(job_id, f"▶ {desc}")
    result = executor.run(cmd, timeout=timeout)
    for ln in (result.stdout or "").splitlines()[-5:]:
        if ln.strip():
            _append_log(job_id, f"  {ln.strip()}")
    if not result.ok:
        for ln in (result.stderr or "").splitlines()[-5:]:
            if ln.strip():
                _append_log(job_id, f"  [err] {ln.strip()}")
    return result


def get_job(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> List[dict]:
    with _LOCK:
        return [{"job_id": j["job_id"], "status": j["status"],
                 "target_id": j["target_id"]} for j in _JOBS.values()]


# ==================== 各平台安装脚本 ====================

def _install_windows(executor: Executor, job_id: str, target: Target) -> str:
    """下载官方预编译 CUDA 包并解压，返回安装后的 engine_path"""
    install_dir = r"C:\llama"
    _run_step(executor, job_id,
              f'powershell -Command "New-Item -ItemType Directory -Force -Path {install_dir} | Out-Null"',
              L("install.mkdir", dir=install_dir))

    # 获取最新 release 的 win-cuda 包下载地址
    get_url_cmd = (
        'powershell -Command "'
        "$ProgressPreference='SilentlyContinue'; "
        "$r=Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'; "
        "$a=$r.assets | Where-Object { $_.name -match 'bin-win-cuda-cu12' -and $_.name -match 'x64' } | Select-Object -First 1; "
        "Write-Output $a.browser_download_url\""
    )
    _append_log(job_id, "▶ " + L("install.query_release"))
    url_res = executor.run(get_url_cmd, timeout=60)
    url = ""
    for ln in url_res.stdout.splitlines():
        if ln.strip().startswith("http"):
            url = ln.strip()
            break
    if not url:
        raise RuntimeError(L("install.err.no_download_url"))
    _append_log(job_id, "  " + L("install.download_source", url=url))

    zip_path = r"C:\llama\llama.zip"
    _run_step(executor, job_id,
              f'powershell -Command "$ProgressPreference=\'SilentlyContinue\'; '
              f'Invoke-WebRequest -Uri \'{url}\' -OutFile \'{zip_path}\'"',
              L("install.download_pkg"), timeout=900)

    _run_step(executor, job_id,
              f'powershell -Command "Expand-Archive -Path \'{zip_path}\' -DestinationPath \'{install_dir}\' -Force"',
              L("install.unzip"))

    # 定位 llama-server.exe（解压后在子目录内）
    find_cmd = (
        f'powershell -Command "Get-ChildItem -Path {install_dir} -Recurse -Filter llama-server.exe '
        '| Select-Object -First 1 -ExpandProperty FullName"'
    )
    fr = executor.run(find_cmd, timeout=30)
    exe_path = ""
    for ln in fr.stdout.splitlines():
        if ln.strip().lower().endswith("llama-server.exe"):
            exe_path = ln.strip()
            break
    if not exe_path:
        raise RuntimeError(L("install.err.no_llama_exe"))
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


def _install_linux(executor: Executor, job_id: str, target: Target) -> str:
    """源码编译 llama.cpp（启用 CUDA 若可用）"""
    _run_step(executor, job_id,
              "command -v cmake && command -v git && command -v g++",
              L("install.check_build_deps"), timeout=30)

    build_dir = "/tmp/llama.cpp"
    _run_step(executor, job_id,
              f"rm -rf {build_dir} && git clone --depth 1 https://github.com/ggml-org/llama.cpp {build_dir}",
              L("install.clone_llama"), timeout=600)

    # 检测是否有 CUDA
    cuda = executor.run("command -v nvcc")
    cmake_flag = "-DGGML_CUDA=ON" if cuda.stdout else ""
    _run_step(executor, job_id,
              f"cd {build_dir} && cmake -B build {cmake_flag} && cmake --build build --config Release -j --target llama-server",
              L("install.compile_llama_cuda") if cmake_flag else L("install.compile_llama"), timeout=2400)

    exe_path = f"{build_dir}/build/bin/llama-server"
    check = executor.run(f'test -f "{exe_path}" && echo FOUND')
    if "FOUND" not in check.stdout:
        raise RuntimeError(L("install.err.no_compile_out"))
    _append_log(job_id, "  " + L("install.engine_path", path=exe_path))
    return exe_path


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
