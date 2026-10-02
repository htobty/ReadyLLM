"""后端界面文案（按请求语言返回）

为什么放在后端：检测结果、安装日志、启停消息都是后端运行时拼出来的，
前端拿不到"未渲染的素材"，只能原样显示。所以这些文案必须有中英两版。

用法：
    from .i18n import L, set_lang
    set_lang(lang)          # 请求入口调一次
    return {"reason": L("detect.no_engine_path")}

要点：
- 语言存在 contextvars 里（请求级隔离）；后台安装线程不继承请求上下文，
  所以 start_install() 会带着 lang 走，并在 worker 开头显式 set_lang。
- 只登记"我们拼的提示"。命令真实输出（pip / git / git clone 的日志）原样透传，
  它们本来就是英文，不该也不需要在目录里。
- 文案里用 {name} 占位，未登记的 key 原样返回 key，便于发现遗漏。
"""

import contextvars

_LANG = contextvars.ContextVar("lang", default="zh")


def norm_lang(lang) -> str:
    """归一化语言标识：en* -> en，其余 -> zh"""
    s = (lang or "").strip().lower()
    return "en" if s.startswith("en") else "zh"


def set_lang(lang):
    return _LANG.set(norm_lang(lang))


def get_lang() -> str:
    return _LANG.get()


def L(key: str, **params) -> str:
    """取当前语言文案并插值"""
    entry = MESSAGES.get(key)
    if not entry:
        return key
    s = entry.get(_LANG.get()) or entry.get("zh") or key
    for k, v in params.items():
        s = s.replace("{" + k + "}", str(v))
    return s


MESSAGES = {
    # ==================== 引擎检测 ====================
    "detect.no_comfyui_dir": {
        "zh": "未配置 ComfyUI 安装目录（engine_path 应指向 ComfyUI 根目录）",
        "en": "ComfyUI install directory not configured (engine_path should point to the ComfyUI root)",
    },
    "detect.comfyui_missing": {
        "zh": "指定目录下未找到 ComfyUI（main.py）",
        "en": "ComfyUI (main.py) not found in the given directory",
    },
    "detect.no_engine_path": {
        "zh": "未配置引擎路径",
        "en": "Engine path not configured",
    },
    "detect.llama_missing": {
        "zh": "指定路径下未找到 llama-server",
        "en": "llama-server not found at the given path",
    },
    "detect.vllm_windows": {
        "zh": "vLLM 不支持 Windows 原生运行，请在 WSL2 (Linux) 中部署，或改用 llama.cpp",
        "en": "vLLM does not run on Windows natively. Deploy it in WSL2 (Linux), or switch to llama.cpp.",
    },
    "detect.vllm_missing": {
        "zh": "未检测到 vllm 命令，请先安装（pip install vllm）",
        "en": "vllm command not found - install it first (pip install vllm)",
    },
    "detect.sglang_windows": {
        "zh": "SGLang 官方安装说明面向 Linux + NVIDIA GPU，请在 WSL2 (Linux) 中部署，或改用 llama.cpp",
        "en": "SGLang's official install guidance targets Linux + NVIDIA GPUs. Deploy it in WSL2 (Linux), or switch to llama.cpp.",
    },
    "detect.sglang_missing": {
        "zh": "未检测到 sglang 命令，请先安装（pip install sglang，需 CUDA 环境）",
        "en": "sglang command not found - install it first (pip install sglang, needs CUDA)",
    },

    # ==================== 安装：进度日志 ====================
    "install.mkdir": {
        "zh": "创建安装目录 {dir}",
        "en": "Creating install directory {dir}",
    },
    "install.query_release": {
        "zh": "查询最新预编译包版本",
        "en": "Looking up the latest prebuilt release",
    },
    "install.download_source": {
        "zh": "下载源: {url}",
        "en": "Download URL: {url}",
    },
    "install.download_pkg": {
        "zh": "下载预编译包（可能较大，请稍候）",
        "en": "Downloading prebuilt package (may be large, please wait)",
    },
    "install.unzip": {
        "zh": "解压安装包",
        "en": "Extracting package",
    },
    "install.engine_path": {
        "zh": "引擎路径: {path}",
        "en": "Engine path: {path}",
    },
    "install.check_pip": {
        "zh": "检查 pip 是否可用",
        "en": "Checking pip availability",
    },
    "install.gpu_detected": {
        "zh": "检测到 GPU/CUDA: {info}",
        "en": "GPU/CUDA detected: {info}",
    },
    "install.no_cuda_sglang": {
        "zh": "未检测到 CUDA，SGLang 需要 NVIDIA GPU 环境，安装后可能无法正常运行",
        "en": "No CUDA detected - SGLang needs an NVIDIA GPU and may not run properly",
    },
    "install.no_cuda_vllm": {
        "zh": "未检测到 CUDA，vLLM 主要面向 NVIDIA GPU，安装后可能无法正常运行",
        "en": "No CUDA detected - vLLM targets NVIDIA GPUs and may not run properly",
    },
    "install.pip_sglang": {
        "zh": "pip 安装 SGLang（体积较大、耗时较长，请稍候）",
        "en": "Installing SGLang with pip (large download, this takes a while)",
    },
    "install.pip_vllm": {
        "zh": "pip 安装 vLLM（体积较大、耗时较长，请稍候）",
        "en": "Installing vLLM with pip (large download, this takes a while)",
    },
    "install.brew_llama": {
        "zh": "通过 Homebrew 安装 llama.cpp",
        "en": "Installing llama.cpp via Homebrew",
    },
    "install.check_build_deps": {
        "zh": "检查编译依赖 (cmake/git/g++)",
        "en": "Checking build dependencies (cmake/git/g++)",
    },
    "install.clone_llama": {
        "zh": "克隆 llama.cpp 源码",
        "en": "Cloning llama.cpp source",
    },
    "install.compile_llama": {
        "zh": "编译 llama-server",
        "en": "Building llama-server",
    },
    "install.compile_llama_cuda": {
        "zh": "编译 llama-server（CUDA）",
        "en": "Building llama-server (CUDA)",
    },
    "install.check_git_python": {
        "zh": "检查 git 与 python",
        "en": "Checking git and python",
    },
    "install.comfyui_exists": {
        "zh": "检测到已存在 ComfyUI，跳过克隆",
        "en": "Existing ComfyUI found, skipping clone",
    },
    "install.clone_comfyui": {
        "zh": "克隆 ComfyUI 仓库",
        "en": "Cloning the ComfyUI repository",
    },
    "install.pip_comfyui": {
        "zh": "pip 安装 ComfyUI 依赖（含 PyTorch，体积大、耗时长，请稍候）",
        "en": "Installing ComfyUI dependencies with pip (includes PyTorch; large and slow)",
    },
    "install.comfyui_dir": {
        "zh": "ComfyUI 目录: {path}",
        "en": "ComfyUI directory: {path}",
    },
    "install.start": {
        "zh": "开始为「{name}」({os}) 安装 {engine}",
        "en": "Starting {engine} installation for \"{name}\" ({os})",
    },
    "install.done": {
        "zh": "✓ 安装完成，引擎路径已自动回填到配置",
        "en": "✓ Installation complete - engine path saved to the machine config",
    },
    "install.failed": {
        "zh": "✗ 安装失败: {err}",
        "en": "✗ Installation failed: {err}",
    },

    # ==================== 安装：错误 ====================
    "install.err.no_download_url": {
        "zh": "无法获取预编译包下载地址（请检查目标机网络或 GitHub 可访问性）",
        "en": "Could not resolve the prebuilt package URL (check the target machine's network access to GitHub)",
    },
    "install.err.no_llama_exe": {
        "zh": "解压后未找到 llama-server.exe",
        "en": "llama-server.exe not found after extraction",
    },
    "install.err.no_sglang_cmd": {
        "zh": "安装完成但未找到 sglang 命令，请检查 pip 输出或 PATH",
        "en": "Install finished but the sglang command is missing - check pip output or PATH",
    },
    "install.err.no_vllm_cmd": {
        "zh": "安装完成但未找到 vllm 命令，请检查 pip 输出或 PATH",
        "en": "Install finished but the vllm command is missing - check pip output or PATH",
    },
    "install.err.no_brew": {
        "zh": "目标机未安装 Homebrew，请先安装 brew (https://brew.sh) 后重试",
        "en": "Homebrew is not installed on the target machine - install it from https://brew.sh and retry",
    },
    "install.err.no_llama_brew": {
        "zh": "安装完成但未找到 llama-server，请检查 brew 输出",
        "en": "Install finished but llama-server is missing - check brew output",
    },
    "install.err.no_compile_out": {
        "zh": "编译完成但未生成 llama-server",
        "en": "Build finished but llama-server was not produced",
    },
    "install.err.no_comfyui_main": {
        "zh": "安装完成但未找到 ComfyUI main.py，请检查克隆/网络",
        "en": "Install finished but ComfyUI main.py is missing - check the clone and network",
    },
    "install.err.vllm_windows": {
        "zh": "vLLM 不支持 Windows 原生运行，请在 WSL2 (Linux) 中安装，或改用 llama.cpp",
        "en": "vLLM does not run on Windows natively - install it in WSL2 (Linux), or use llama.cpp",
    },
    "install.err.sglang_windows": {
        "zh": "SGLang 官方安装说明面向 Linux + NVIDIA GPU，请在 WSL2 (Linux) 中安装，或改用 llama.cpp",
        "en": "SGLang's official install guidance targets Linux + NVIDIA GPUs - install it in WSL2 (Linux), or use llama.cpp",
    },

    # ==================== 目标机 API ====================
    "target.not_found": {
        "zh": "目标机器不存在",
        "en": "Target machine not found",
    },
    "target.not_found_hint": {
        "zh": "目标机器不存在，请先在设置中配置",
        "en": "Target machine not found - configure one in Settings first",
    },
    "target.conn_ok": {
        "zh": "连接成功",
        "en": "Connected",
    },
    "target.conn_fail": {
        "zh": "连接失败: {err}",
        "en": "Connection failed: {err}",
    },
    "target.conn_error": {
        "zh": "连接异常: {err}",
        "en": "Connection error: {err}",
    },

    # ==================== 引擎启停 ====================
    "engine.start_sent": {
        "zh": "启动命令已发送",
        "en": "Start command sent",
    },
    "engine.start_sent_weights": {
        "zh": "启动命令已发送（首次加载模型需下载权重，请耐心等待）",
        "en": "Start command sent (the first run downloads weights, please wait)",
    },
    "engine.start_sent_comfyui": {
        "zh": "ComfyUI 启动命令已发送（首次启动需加载依赖，请耐心等待）",
        "en": "ComfyUI start command sent (the first launch loads dependencies, please wait)",
    },
    "engine.write_script_fail": {
        "zh": "写入启动脚本失败: {err}",
        "en": "Failed to write the launch script: {err}",
    },
    "engine.start_fail": {
        "zh": "启动失败: {err}",
        "en": "Start failed: {err}",
    },
    "engine.stop_ok": {
        "zh": "服务已停止",
        "en": "Service stopped",
    },
    "engine.stop_result": {
        "zh": "停止结果: {err}",
        "en": "Stop result: {err}",
    },

    # ==================== ComfyUI 服务 ====================
    "comfyui.workflow_write_fail": {
        "zh": "写入 workflow 临时文件失败",
        "en": "Failed to write the temporary workflow file",
    },
    "comfyui.workflow_rejected": {
        "zh": "workflow 被拒绝: {err}",
        "en": "Workflow rejected: {err}",
    },
    "comfyui.submit_unparsable": {
        "zh": "提交失败，响应无法解析: {err}",
        "en": "Submit failed - could not parse the response: {err}",
    },
    "comfyui.exec_error": {
        "zh": "ComfyUI 执行报错",
        "en": "ComfyUI reported an execution error",
    },

    # ==================== 部署页 ====================
    "deploy.no_models_dir": {
        "zh": "未配置模型目录",
        "en": "Model directory not configured",
    },
    "deploy.hf_weights": {
        "zh": "{engine} 加载 HuggingFace 权重（非 GGUF），请直接填写模型 ID 或本地权重目录",
        "en": "{engine} loads HuggingFace weights (not GGUF) - enter a model ID or a local weights directory",
    },
    "deploy.engine_no_video": {
        "zh": "当前目标机引擎不支持视频生成，请改用 ComfyUI",
        "en": "The current engine does not support video generation - switch to ComfyUI",
    },
    "deploy.engine_no_upscale": {
        "zh": "当前引擎不支持超分，请改用 ComfyUI",
        "en": "The current engine does not support upscaling - switch to ComfyUI",
    },
    "deploy.comfyui_not_running": {
        "zh": "ComfyUI 服务未运行，请先在部署页启动",
        "en": "ComfyUI is not running - start it from the Deploy page first",
    },
    "deploy.image_not_exist": {
        "zh": "首帧图不存在: {path}",
        "en": "First-frame image not found: {path}",
    },
    "deploy.image_read_fail": {
        "zh": "读取首帧图失败: {err}",
        "en": "Failed to read the first-frame image: {err}",
    },
    "deploy.image_upload_fail": {
        "zh": "首帧图上传到目标机 input 目录失败",
        "en": "Failed to upload the first-frame image to the target's input directory",
    },
    "deploy.ref_not_exist": {
        "zh": "参考图不存在: {path}",
        "en": "Reference image not found: {path}",
    },
    "deploy.ref_read_fail": {
        "zh": "读取参考图失败: {err}",
        "en": "Failed to read the reference image: {err}",
    },
    "deploy.ref_upload_fail": {
        "zh": "参考图上传失败: {path}",
        "en": "Failed to upload the reference image: {path}",
    },
    "deploy.pic_not_exist": {
        "zh": "图片不存在: {path}",
        "en": "Image not found: {path}",
    },
    "deploy.pic_read_fail": {
        "zh": "读取图片失败: {err}",
        "en": "Failed to read the image: {err}",
    },
    "deploy.pic_upload_fail": {
        "zh": "图片上传到目标机 input 目录失败",
        "en": "Failed to upload the image to the target's input directory",
    },
    "deploy.need_image": {
        "zh": "需提供 image_path 或 image_name",
        "en": "Provide either image_path or image_name",
    },
    "deploy.task_submitted": {
        "zh": "生成任务已提交",
        "en": "Generation task submitted",
    },
    "deploy.upscale_submitted": {
        "zh": "超分任务已提交",
        "en": "Upscaling task submitted",
    },
    "deploy.no_progress_support": {
        "zh": "当前引擎不支持生成任务查询",
        "en": "The current engine does not support generation progress queries",
    },
    "deploy.params_format_error": {
        "zh": "参数格式错误: {err}",
        "en": "Invalid parameter format: {err}",
    },
    "deploy.storyboard_fail": {
        "zh": "分镜生成失败：请确认已在设置中配置可用的大模型 API",
        "en": "Storyboard generation failed - make sure a working LLM API is configured in Settings",
    },
    "deploy.engine_generic_args": {
        "zh": "{engine} 使用引擎通用默认参数（该引擎暂不支持自动调优）",
        "en": "{engine} uses the engine's generic default parameters (auto-tuning is not supported for this engine yet)",
    },
    "deploy.no_comfyui_dir": {
        "zh": "未配置 ComfyUI 目录",
        "en": "ComfyUI directory not configured",
    },
    "deploy.illegal_path": {
        "zh": "非法文件路径",
        "en": "Invalid file path",
    },
    "deploy.clip_not_found": {
        "zh": "成片文件不存在或读取失败",
        "en": "Output video not found or could not be read",
    },

    # ==================== 调优：启动前校验 ====================

    "tune.err.no_target": {
        "zh": "目标机器不存在",
        "en": "Target machine not found",
    },
    "tune.err.no_engine": {
        "zh": "未配置推理引擎，请先在设置中安装",
        "en": "Inference engine not configured - install one in Settings first",
    },
    "tune.err.llama_only": {
        "zh": "自动调优目前仅支持 llama.cpp 引擎（vLLM 参数体系不同，暂不支持）",
        "en": "Auto-tuning currently supports the llama.cpp engine only (vLLM uses a different parameter set)",
    },
    "tune.err.no_model": {
        "zh": "未选择模型或模型目录为空",
        "en": "No model selected, or the model directory is empty",
    },
    "tune.err.ctx_small": {
        "zh": "ctx-size 过小，请至少 1024",
        "en": "ctx-size is too small - use at least 1024",
    },

    # ==================== 调优：优化目标 ====================

    "tune.goal.latency": {
        "zh": "端到端体感",
        "en": "End-to-end feel",
    },
    "tune.goal.throughput": {
        "zh": "解码吞吐",
        "en": "Decode throughput",
    },
    "tune.goal.prefill": {
        "zh": "长文本预填充",
        "en": "Long-context prefill",
    },

    # ==================== 自动调优：日志 ====================

    "tune.log.no_engine": {
        "zh": "✗ 未检测到推理引擎",
        "en": "✗ Inference engine not detected",
    },
    "tune.log.hw": {
        "zh": "目标机显存: {vram} GB | 模型: {model} ({size} GB) | ctx 固定 {ctx} | 目标: {goal}",
        "en": "Target VRAM: {vram} GB | Model: {model} ({size} GB) | ctx fixed at {ctx} | Goal: {goal}",
    },
    "tune.log.baseline_test": {
        "zh": "【基线】测试你当前配置",
        "en": "[Baseline] Testing your current configuration",
    },
    "tune.log.baseline_score": {
        "zh": "  基线分: {score}",
        "en": "  Baseline score: {score}",
    },
    "tune.log.coarse_start": {
        "zh": "【阶段1 coarse】{n} 组主导因素组合",
        "en": "[Stage 1 coarse] {n} dominant-factor combinations",
    },
    "tune.log.coarse_skip": {
        "zh": "  跳过(显存不足): {cfg}",
        "en": "  Skipped (insufficient VRAM): {cfg}",
    },
    "tune.log.coarse_cpu": {
        "zh": "  全 GPU 组合均超显存，降级用 CPU 兜底(n-gpu-layers=0)",
        "en": "  All GPU combinations exceed VRAM; falling back to CPU (n-gpu-layers=0)",
    },
    "tune.log.coarse_best": {
        "zh": "  coarse 最优: {label} (分 {score})",
        "en": "  coarse best: {label} (score {score})",
    },
    "tune.log.fine_start": {
        "zh": "【阶段2 fine】坐标下降，调 {params}",
        "en": "[Stage 2 fine] Coordinate descent over {params}",
    },
    "tune.log.fine_improved": {
        "zh": "    ✓ 改善: {param}={value} 分→{score}",
        "en": "    ✓ Improved: {param}={value} score → {score}",
    },
    "tune.log.fine_done": {
        "zh": "  fine 收敛: {label} (分 {score})",
        "en": "  fine converged: {label} (score {score})",
    },
    "tune.log.run_fail": {
        "zh": "  [{tag}] {label} 启动失败: {err}",
        "en": "  [{tag}] {label} failed to start: {err}",
    },
    "tune.log.run_timeout": {
        "zh": "  [{tag}] {label} 启动超时(可能显存不足)",
        "en": "  [{tag}] {label} startup timed out (possibly out of VRAM)",
    },
    "tune.log.run_result": {
        "zh": "  [{tag}] {label} → 解码{decode} t/s, 预填充{prefill} t/s, GPU {gpu}%",
        "en": "  [{tag}] {label} → decode {decode} t/s, prefill {prefill} t/s, GPU {gpu}%",
    },
    "tune.log.done": {
        "zh": "✓ 调优完成，推荐: {label} (分 {score})",
        "en": "✓ Tuning complete, recommended: {label} (score {score})",
    },
    "tune.log.save_fail": {
        "zh": "  ⚠ 调优结果落盘失败: {err}",
        "en": "  ⚠ Failed to persist the tuning result: {err}",
    },
    "tune.log.exception": {
        "zh": "✗ 异常: {err}",
        "en": "✗ Error: {err}",
    },
    "tune.fail.engine_missing": {
        "zh": "目标机未检测到推理引擎，请先一键安装",
        "en": "No inference engine detected on the target machine - install one first",
    },
    "tune.fail.no_config": {
        "zh": "coarse 阶段无可用配置（可能显存不足）",
        "en": "No usable configuration in the coarse stage (possibly insufficient VRAM)",
    },

    # ==================== AI 调优：启动前校验 ====================

    "tune.ai.err.no_engine": {
        "zh": "未配置推理引擎",
        "en": "Inference engine not configured",
    },
    "tune.ai.err.llama_only": {
        "zh": "AI 调优目前仅支持 llama.cpp 引擎（vLLM 参数体系不同，暂不支持）",
        "en": "AI tuning currently supports the llama.cpp engine only (vLLM uses a different parameter set)",
    },
    "tune.ai.err.no_api": {
        "zh": "未配置 AI API，请先在设置中配置",
        "en": "AI API not configured - set it up in Settings first",
    },

    # ==================== AI 调优：日志 ====================

    "tune.ai.log.no_engine": {
        "zh": "目标机未检测到推理引擎",
        "en": "No inference engine detected on the target machine",
    },
    "tune.ai.log.collect_hw": {
        "zh": "采集硬件信息...",
        "en": "Collecting hardware info...",
    },
    "tune.ai.log.hw": {
        "zh": "硬件: {gpu} | 模型: {model} ({size}GB) | ctx: {ctx}",
        "en": "Hardware: {gpu} | Model: {model} ({size}GB) | ctx: {ctx}",
    },
    "tune.ai.log.baseline_from_history": {
        "zh": "采用上次调优结果作为基线（{src}，实测 {score} t/s，{ts}）",
        "en": "Using the previous tuning result as baseline ({src}, measured {score} t/s, {ts})",
    },
    "tune.ai.log.params": {
        "zh": "  参数: {params}",
        "en": "  Params: {params}",
    },
    "tune.ai.log.gen_config": {
        "zh": "生成确定性基础配置...",
        "en": "Generating the deterministic base configuration...",
    },
    "tune.ai.log.baseline_test": {
        "zh": "实测基线配置...",
        "en": "Benchmarking the baseline configuration...",
    },
    "tune.ai.log.baseline_ok": {
        "zh": "  ✓ 基线实测: 解码 {decode} t/s | 预填充 {prefill} t/s | GPU {gpu}%",
        "en": "  ✓ Baseline measured: decode {decode} t/s | prefill {prefill} t/s | GPU {gpu}%",
    },
    "tune.ai.log.baseline_fail": {
        "zh": "  ⚠ 基线实测失败，AI 将从零开始",
        "en": "  ⚠ Baseline benchmark failed; the AI will start from scratch",
    },
    "tune.ai.log.round": {
        "zh": "【第 {n}/{max} 轮】调用 AI 分析...",
        "en": "[Round {n}/{max}] Asking the AI to analyze...",
    },
    "tune.ai.log.parse_fail": {
        "zh": "  ⚠ AI 返回无法解析，原始内容: {raw}",
        "en": "  ⚠ Could not parse the AI response; raw content: {raw}",
    },
    "tune.ai.log.ai_reasoning": {
        "zh": "  AI 分析: {text}",
        "en": "  AI analysis: {text}",
    },
    "tune.ai.log.ai_done": {
        "zh": "  ✓ AI 认为已找到最优 (置信度: {conf})",
        "en": "  ✓ The AI considers this optimal (confidence: {conf})",
    },
    "tune.ai.log.ai_final": {
        "zh": "✓ AI 调优完成，推荐参数: {params}",
        "en": "✓ AI tuning complete, recommended parameters: {params}",
    },
    "tune.ai.log.unknown_action": {
        "zh": "  ⚠ 未知 action: {action}，要求 AI 重试",
        "en": "  ⚠ Unknown action: {action}; asking the AI to retry",
    },
    "tune.ai.log.test_params": {
        "zh": "  测试参数: {params}",
        "en": "  Testing parameters: {params}",
    },
    "tune.ai.log.test_fail": {
        "zh": "  ✗ 测试失败（启动超时或异常）",
        "en": "  ✗ Benchmark failed (startup timeout or error)",
    },
    "tune.ai.log.result": {
        "zh": "  结果: 解码 {decode} t/s | 预填充 {prefill} t/s | GPU {gpu}% | 显存 {vram}% | CPU {cpu}% | 内存 {mem}/{memtotal}GB",
        "en": "  Result: decode {decode} t/s | prefill {prefill} t/s | GPU {gpu}% | VRAM {vram}% | CPU {cpu}% | RAM {mem}/{memtotal}GB",
    },
    "tune.ai.log.max_rounds": {
        "zh": "达到最大轮次 {max}，使用历史最佳结果",
        "en": "Reached the maximum of {max} rounds; using the best historical result",
    },
    "tune.ai.log.start_fail": {
        "zh": "  启动失败: {err}",
        "en": "  Failed to start: {err}",
    },
    "tune.ai.log.start_timeout": {
        "zh": "  启动超时",
        "en": "  Startup timed out",
    },
    "tune.fail.all_rounds": {
        "zh": "所有轮次均失败",
        "en": "All rounds failed",
    },
    "tune.fail.round_llm": {
        "zh": "第 {n} 轮 LLM 调用失败（原因见上方日志）",
        "en": "LLM call failed in round {n} (see the log above for the reason)",
    },

    # ==================== AI 调优：LLM 调用 ====================

    "tune.ai.llm.no_url": {
        "zh": "  LLM 失败: API 地址为空，请先在 AI 调优设置里填写 api_url",
        "en": "  LLM failed: API URL is empty - fill in api_url in the AI tuning settings first",
    },
    "tune.ai.llm.request": {
        "zh": "  → 请求 LLM: {url} | model={model}",
        "en": "  → LLM request: {url} | model={model}",
    },
    "tune.ai.llm.no_key": {
        "zh": "  ⚠ 未配置 API Key（若服务需要鉴权会返回 401）",
        "en": "  ⚠ No API key configured (a 401 will come back if the service requires auth)",
    },
    "tune.ai.llm.no_choices": {
        "zh": "  LLM 返回无 choices，原始响应: {raw}",
        "en": "  LLM response has no choices; raw response: {raw}",
    },
    "tune.ai.llm.http_error": {
        "zh": "  LLM HTTP {code} 错误: {body}",
        "en": "  LLM HTTP {code} error: {body}",
    },
    "tune.ai.llm.net_error": {
        "zh": "  LLM 网络错误（地址不通/超时/DNS）: {err}",
        "en": "  LLM network error (unreachable address / timeout / DNS): {err}",
    },
    "tune.ai.llm.exception": {
        "zh": "  LLM 调用异常: {err}",
        "en": "  LLM call raised an exception: {err}",
    },

    # ==================== AI 调优：API 连通性测试 ====================

    "tune.ai.conn.empty_url": {
        "zh": "API 地址为空",
        "en": "API URL is empty",
    },
    "tune.ai.conn.ok": {
        "zh": "连接成功，可用模型: {models}",
        "en": "Connected. Available models: {models}",
    },
    "tune.ai.conn.auth_fail": {
        "zh": "认证失败 (HTTP {code})，请检查 API Key",
        "en": "Authentication failed (HTTP {code}) - check the API key",
    },
    "tune.ai.conn.reachable": {
        "zh": "服务可达 (HTTP {code})，但无法列出模型",
        "en": "Service reachable (HTTP {code}), but the model list is unavailable",
    },
    "tune.ai.conn.fail": {
        "zh": "连接失败: {err}",
        "en": "Connection failed: {err}",
    },

    # ==================== 调优：来源标签与结论 ====================

    "tune.src.tuner": {
        "zh": "自动调优",
        "en": "Auto tuning",
    },
    "tune.src.ai": {
        "zh": "AI 调优",
        "en": "AI tuning",
    },
    "tune.reason.deterministic": {
        "zh": "确定性生成器输出（非 AI）",
        "en": "Deterministic generator output (not AI)",
    },
    "tune.reason.max_rounds": {
        "zh": "达到最大轮次，取历史最佳（第 {n} 轮）",
        "en": "Reached the maximum rounds; using the best historical result (round {n})",
    },

    # ==================== 调优结果保存 ====================

    "tune.save.empty": {
        "zh": "无参数可保存",
        "en": "No parameters to save",
    },
    "tune.save.ok": {
        "zh": "已保存到 {model} 的部署参数（含 ctx={ctx}）",
        "en": "Saved to the deploy parameters for {model} (with ctx={ctx})",
    },

    # ==================== 确定性配置生成器 ====================
    # 这些 reasoning/warnings 会逐条写进 AI 调优日志面板，属于用户可见文案

    "cfg.gen.layers": {
        "zh": "推断模型层数: {n}（基于文件名和大小）",
        "en": "Inferred layer count: {n} (from the filename and file size)",
    },
    "cfg.gen.fits": {
        "zh": "模型 {size}GB < 可用显存 {avail}GB ({vram}×{headroom}) → 全量 GPU 卸载",
        "en": "Model {size}GB < usable VRAM {avail}GB ({vram}×{headroom}) → full GPU offload",
    },
    "cfg.gen.over": {
        "zh": "模型 {size}GB 超过可用显存 {avail}GB，只能卸载 {fit}/{total} 层到 GPU，性能会显著下降",
        "en": "Model {size}GB exceeds usable VRAM {avail}GB - only {fit}/{total} layers fit on the GPU, with a significant performance drop",
    },
    "cfg.gen.partial": {
        "zh": "模型放不下 → 部分卸载 {fit} 层（这是唯一允许非 all 的情况）",
        "en": "Model does not fit → partial offload of {fit} layers (the only case where non-'all' is allowed)",
    },
    "cfg.gen.mtp_no_room": {
        "zh": "显存余量不足以同时容纳 draft 模型，跳过投机解码",
        "en": "Not enough VRAM headroom for the draft model - skipping speculative decoding",
    },
    "cfg.gen.mtp_fallback": {
        "zh": "显存不足以启用投机解码，回退到无投机方案",
        "en": "Insufficient VRAM for speculative decoding - falling back to no speculation",
    },
    "cfg.gen.mtp_cache": {
        "zh": "模型支持 MTP，为保证投机解码选用 cache={cache}（KV 占 {kv}GB + draft {draft}GB，剩余 {left}GB）",
        "en": "Model supports MTP; using cache={cache} to keep speculative decoding (KV takes {kv}GB + draft {draft}GB, {left}GB left)",
    },
    "cfg.gen.no_mtp": {
        "zh": "模型不支持 MTP 投机解码（文件名未检测到相关标记）",
        "en": "Model does not support MTP speculative decoding (no marker found in the filename)",
    },
    "cfg.gen.kv": {
        "zh": "剩余显存 {left}GB，ctx={ctx}，KV cache({cache}) 约占 {kv}GB",
        "en": "Remaining VRAM {left}GB, ctx={ctx}, KV cache ({cache}) takes about {kv}GB",
    },
    "cfg.gen.batch": {
        "zh": "扣除模型+KV+draft后剩余 {left}GB → batch={batch}, ubatch={ubatch}",
        "en": "{left}GB left after model + KV + draft → batch={batch}, ubatch={ubatch}",
    },
    "cfg.gen.threads": {
        "zh": "CPU {cores} 核 → threads={threads}",
        "en": "CPU has {cores} cores → threads={threads}",
    },
}