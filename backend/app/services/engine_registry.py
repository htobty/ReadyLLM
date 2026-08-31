"""推理引擎注册表 / 工厂

集中管理所有可用引擎的元信息与适配器实例化。调用方（deploy / installer /
前端配置页）只需按 target.engine_type 取适配器或元信息，不再硬编码具体类。

新增引擎步骤：
  1. 实现一个 EngineAdapter 子类（如 vllm.VLLMAdapter）
  2. 在下方 _ADAPTERS 与 ENGINE_META 各加一条
"""

from typing import Optional

from .engine_adapter import EngineAdapter, StartParams  # noqa: F401  (re-export)
from .llama_cpp import LlamaCppAdapter
from .vllm import VLLMAdapter
from .comfyui import ComfyUIAdapter
from ..models.target import Target

# engine_type -> 适配器类
_ADAPTERS = {
    "llama_cpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "comfyui": ComfyUIAdapter,
}

# engine_type -> 展示与能力元信息（前端配置页/部署页消费）
ENGINE_META = {
    "llama_cpp": {
        "label": "llama.cpp",
        "desc": "原生跨平台，支持 GGUF 量化模型，CPU/GPU 混合推理，普通用户首选",
        "supported_os": ["windows", "linux", "macos"],
        "model_format": "gguf",
        "install_hint": "一键安装官方预编译包 / brew / 源码编译",
        "default_cmd": "llama-server",
    },
    "vllm": {
        "label": "vLLM",
        "desc": "高吞吐推理引擎，需 NVIDIA GPU + CUDA，使用 HuggingFace safetensors 权重",
        "supported_os": ["linux", "macos"],
        "model_format": "safetensors",
        "install_hint": "pip install vllm（需 Python 3.9+ 与 CUDA 环境）",
        "default_cmd": "vllm",
        # vLLM 不支持 Windows 原生，前端据此提示走 WSL2
        "windows_note": "vLLM 不支持 Windows 原生运行，请在 WSL2 (Linux) 中部署，或改用 llama.cpp",
    },
    "comfyui": {
        "label": "ComfyUI",
        "desc": "节点式图像/视频生成引擎，本地部署开源视频模型（Wan2.1 / CogVideoX / LTX-Video 等），需 NVIDIA GPU 与大显存",
        "supported_os": ["windows", "linux", "macos"],
        "model_format": "safetensors",
        "task": "video",
        "install_hint": "git clone ComfyUI + pip install -r requirements.txt（需 Python 3.10+ 与 CUDA）",
        "default_cmd": "python main.py",
        "default_port": 8188,
        "note": "视频生成模型显存需求高（量化版 6~24GB 不等），请按本机显存在商店筛选可跑的模型",
    },
}


def get_adapter(executor, target: Target) -> EngineAdapter:
    """按 target.engine_type 返回对应引擎适配器实例"""
    cls = _ADAPTERS.get(target.engine_type, LlamaCppAdapter)
    return cls(executor, target)


def get_meta(engine_type: str) -> Optional[dict]:
    """返回引擎元信息；未知类型返回 None"""
    return ENGINE_META.get(engine_type)


def is_supported_on(engine_type: str, os_name: str) -> bool:
    """该引擎是否支持给定目标 OS"""
    meta = ENGINE_META.get(engine_type)
    if not meta:
        return False
    return os_name in meta["supported_os"]


def list_engines() -> list:
    """列出全部引擎（含 type 字段），供前端下拉/卡片渲染"""
    out = []
    for et, meta in ENGINE_META.items():
        out.append({"type": et, **meta})
    return out
