"""模型目录服务

支持两种模式：
  1. 内置精选目录（硬编码，保证离线可用）
  2. 动态获取（从 HuggingFace API 拉取热门 GGUF 模型，支持手动刷新）

每个条目提供多源下载信息：
  - huggingface：原始站（境外，可能慢/不可达）
  - hf-mirror：HuggingFace 镜像站（境内友好，路径与 HF 完全一致，默认）
  - modelscope：魔搭社区（境内最快，需该模型有对应 ms_repo，否则回退镜像站）

注意：模型大小为近似值，仅用于展示与显存筛选，实际以仓库为准。
"""

import json
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import List, Optional

HF_BASE = "https://huggingface.co"
HF_MIRROR_BASE = "https://hf-mirror.com"
MS_BASE = "https://modelscope.cn/models"

# 源优先级：默认镜像站（最稳，路径与 HF 一致）
DEFAULT_SOURCE = "hf-mirror"
SOURCE_LABELS = {
    "huggingface": "HuggingFace",
    "hf-mirror": "HF 镜像站",
    "modelscope": "魔搭 ModelScope",
}


@dataclass
class ModelEntry:
    id: str
    name: str            # 展示名，如 "Qwen3-8B"
    quant: str           # 量化，如 Q4_K_M
    repo: str            # HuggingFace repo，如 "bartowski/Qwen2.5-8B-Instruct-GGUF"
    filename: str        # 文件名，如 "Qwen2.5-8B-Instruct-Q4_K_M.gguf"
    size_gb: float       # 近似大小
    min_vram_gb: int     # 推荐最低显存
    desc: str = ""
    tags: List[str] = field(default_factory=list)
    ms_repo: str = ""    # 魔搭仓库（如 "Qwen/Qwen3-8B-GGUF"），留空则该模型不支持魔搭源
    category: str = "text"  # 模型类别：text=文本推理(llama.cpp/vLLM)，video=视频生成(ComfyUI)
    engine: str = ""        # 推荐引擎：留空表示按 category 推断（text->llama_cpp，video->comfyui）

    def hf_url(self, base: str = HF_BASE) -> str:
        return f"{base}/{self.repo}/resolve/main/{self.filename}"

    def ms_url(self) -> str:
        # 魔搭 GGUF 直链：resolve/master，部分仓库用文件名
        return f"{MS_BASE}/{self.ms_repo}/resolve/master/{self.filename}"

    def resolve(self, source: str = DEFAULT_SOURCE) -> tuple:
        """按源解析下载直链，返回 (url, 实际使用的源)。
        魔搭无对应仓库时回退镜像站。"""
        if source == "modelscope" and self.ms_repo:
            return self.ms_url(), "modelscope"
        if source == "huggingface":
            return self.hf_url(HF_BASE), "huggingface"
        # hf-mirror，或 modelscope 不可用时的回退
        return self.hf_url(HF_MIRROR_BASE), "hf-mirror"

    def available_sources(self) -> List[str]:
        srcs = ["hf-mirror", "huggingface"]
        if self.ms_repo:
            srcs.insert(0, "modelscope")
        return srcs

    def to_dict(self, source: str = DEFAULT_SOURCE) -> dict:
        d = asdict(self)
        url, used = self.resolve(source)
        d["download_url"] = url
        d["used_source"] = used
        d["available_sources"] = self.available_sources()
        return d


# ==================== 精选模型（2025-2026 最新） ====================

CATALOG: List[ModelEntry] = [
    # --- Qwen3.8 系列（最新旗舰，原生 MTP 投机解码） ---
    ModelEntry("qwen38-27b", "Qwen3.8-27B", "IQ4_NL",
               "bartowski/Qwen3.8-27B-GGUF", "Qwen3.8-27B-IQ4_NL.gguf",
               15.2, 20, "最新旗舰，原生 MTP 投机解码，24G 显存流畅", ["qwen3.8", "mtp", "large"],
               ms_repo="Qwen/Qwen3.8-27B-GGUF"),
    ModelEntry("qwen38-27b-q4km", "Qwen3.8-27B", "Q4_K_M",
               "bartowski/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf",
               16.5, 22, "旗舰标准量化，兼容性好", ["qwen3.8", "mtp", "large"],
               ms_repo="Qwen/Qwen3.8-27B-GGUF"),
    ModelEntry("qwen38-8b", "Qwen3.8-8B", "Q4_K_M",
               "bartowski/Qwen3.8-8B-GGUF", "Qwen3.8-8B-Q4_K_M.gguf",
               5.2, 8, "最新小旗舰，MTP 加速，8G 显存流畅", ["qwen3.8", "mtp"],
               ms_repo="Qwen/Qwen3.8-8B-GGUF"),

    # --- Qwen3.5 系列 ---
    ModelEntry("qwen35-32b", "Qwen3.5-32B", "Q4_K_M",
               "bartowski/Qwen3.5-32B-GGUF", "Qwen3.5-32B-Q4_K_M.gguf",
               19.5, 24, "强推理，需 24G 显存", ["qwen3.5", "large"],
               ms_repo="Qwen/Qwen3.5-32B-GGUF"),
    ModelEntry("qwen35-14b", "Qwen3.5-14B", "Q4_K_M",
               "bartowski/Qwen3.5-14B-GGUF", "Qwen3.5-14B-Q4_K_M.gguf",
               9.2, 12, "平衡能力与速度", ["qwen3.5"],
               ms_repo="Qwen/Qwen3.5-14B-GGUF"),
    ModelEntry("qwen35-8b", "Qwen3.5-8B", "Q4_K_M",
               "bartowski/Qwen3.5-8B-GGUF", "Qwen3.5-8B-Q4_K_M.gguf",
               5.0, 8, "性价比之选", ["qwen3.5"],
               ms_repo="Qwen/Qwen3.5-8B-GGUF"),
    ModelEntry("qwen35-4b", "Qwen3.5-4B", "Q4_K_M",
               "bartowski/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf",
               2.6, 4, "轻量高效，CPU 也能跑", ["qwen3.5", "small"],
               ms_repo="Qwen/Qwen3.5-4B-GGUF"),

    # --- Llama 4 系列 ---
    ModelEntry("llama4-scout-17b", "Llama-4-Scout-17B", "Q4_K_M",
               "bartowski/Llama-4-Scout-17B-16E-Instruct-GGUF",
               "Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf",
               11.0, 16, "MoE 架构，17B 激活参数，多模态", ["llama4", "moe"],
               ms_repo="LLM-Research/Llama-4-Scout-17B-16E-Instruct-GGUF"),
    ModelEntry("llama4-maverick-17b", "Llama-4-Maverick-17B", "Q4_K_M",
               "bartowski/Llama-4-Maverick-17B-128E-Instruct-GGUF",
               "Llama-4-Maverick-17B-128E-Instruct-Q4_K_M.gguf",
               65.0, 80, "MoE 旗舰，128 专家，需大显存", ["llama4", "moe", "large"],
               ms_repo="LLM-Research/Llama-4-Maverick-17B-128E-Instruct-GGUF"),

    # --- DeepSeek 系列 ---
    ModelEntry("deepseek-r1-distill-32b", "DeepSeek-R1-Distill-32B", "Q4_K_M",
               "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
               "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
               19.5, 24, "推理增强，数学/代码强", ["deepseek", "reasoning", "large"],
               ms_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-GGUF"),
    ModelEntry("deepseek-r1-distill-14b", "DeepSeek-R1-Distill-14B", "Q4_K_M",
               "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
               "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
               9.0, 12, "轻量推理模型", ["deepseek", "reasoning"],
               ms_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF"),
    ModelEntry("deepseek-r1-distill-8b", "DeepSeek-R1-Distill-8B", "Q4_K_M",
               "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
               "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
               5.0, 8, "入门推理模型", ["deepseek", "reasoning"],
               ms_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF"),

    # --- Gemma 3 系列 ---
    ModelEntry("gemma3-27b", "Gemma-3-27B", "Q4_K_M",
               "bartowski/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf",
               16.5, 20, "Google 多模态，视觉+文本", ["gemma3", "multimodal", "large"],
               ms_repo="google/gemma-3-27b-it-GGUF"),
    ModelEntry("gemma3-12b", "Gemma-3-12B", "Q4_K_M",
               "bartowski/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf",
               7.5, 10, "多模态平衡之选", ["gemma3", "multimodal"],
               ms_repo="google/gemma-3-12b-it-GGUF"),
    ModelEntry("gemma3-4b", "Gemma-3-4B", "Q4_K_M",
               "bartowski/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf",
               2.8, 4, "轻量多模态", ["gemma3", "multimodal", "small"],
               ms_repo="google/gemma-3-4b-it-GGUF"),

    # --- Mistral 系列 ---
    ModelEntry("mistral-small-3.2", "Mistral-Small-3.2-24B", "Q4_K_M",
               "bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
               "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
               14.0, 18, "欧洲最强开源，多语言", ["mistral", "multilingual"],
               ms_repo="mistralai/Mistral-Small-3.2-24B-Instruct-2506-GGUF"),

    # --- 代码模型 ---
    ModelEntry("qwen3-coder-32b", "Qwen3-Coder-32B", "Q4_K_M",
               "bartowski/Qwen3-Coder-32B-GGUF", "Qwen3-Coder-32B-Q4_K_M.gguf",
               19.5, 24, "最强开源代码模型", ["code", "qwen3", "large"],
               ms_repo="Qwen/Qwen3-Coder-32B-GGUF"),
    ModelEntry("qwen3-coder-8b", "Qwen3-Coder-8B", "Q4_K_M",
               "bartowski/Qwen3-Coder-8B-GGUF", "Qwen3-Coder-8B-Q4_K_M.gguf",
               5.0, 8, "轻量代码模型", ["code", "qwen3"],
               ms_repo="Qwen/Qwen3-Coder-8B-GGUF"),

    # --- Embedding / Reranker ---
    ModelEntry("bge-m3", "bge-m3", "F16",
               "BAAI/bge-m3-GGUF", "bge-m3-F16.gguf",
               2.2, 4, "多语言向量模型，RAG 必备", ["embedding"],
               ms_repo="BAAI/bge-m3-GGUF"),

    # ==================== 视频生成模型（ComfyUI / safetensors） ====================
    # 说明：视频模型走 ComfyUI 引擎，权重为 safetensors（非 GGUF），显存需求高。
    # filename 这里填 ComfyUI checkpoints 目录下的主权重文件名，下载后需放入
    # ComfyUI/models/checkpoints（端到端验证 step_7 时按实际仓库文件名校准）。
    ModelEntry("wan21-t2v-1.3b", "Wan2.1-T2V-1.3B", "fp16",
               "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
               "split_files/v1/wan2.1_t2v_1.3b_fp16.safetensors",
               6.0, 8, "轻量文生视频，8G 显存可跑，480p 起步",
               ["video", "wan", "t2v"], category="video", engine="comfyui"),
    ModelEntry("wan21-t2v-14b", "Wan2.1-T2V-14B", "fp16",
               "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
               "split_files/v1/wan2.1_t2v_14b_fp16.safetensors",
               28.0, 24, "高质量文生视频旗舰，需 24G+ 显存",
               ["video", "wan", "t2v", "large"], category="video", engine="comfyui"),
    ModelEntry("ltx-video-2b", "LTX-Video-2B", "fp16",
               "Lightricks/LTX-Video", "ltx-video-2b-v0.9.5.safetensors",
               2.5, 6, "极速生成，6G 显存可跑，适合快速预览",
               ["video", "ltx", "fast"], category="video", engine="comfyui"),
    ModelEntry("cogvideox-5b", "CogVideoX-5B", "fp16",
               "zai-org/CogVideoX-5b", "CogVideoX-Fun-V1.1-5b-InP.safetensors",
               11.0, 16, "智谱开源视频模型，1280×720，需 16G 显存",
               ["video", "cogvideo"], category="video", engine="comfyui"),
]


# ==================== 动态获取（HuggingFace API） ====================

# 优先用镜像站 API（国内可达），失败时回退 HuggingFace 原始站
_HF_API_MIRROR = "https://hf-mirror.com/api/models"
_HF_API_ORIGIN = "https://huggingface.co/api/models"
_dynamic_cache: Optional[List[dict]] = None
_dynamic_cache_time: float = 0
_DYNAMIC_CACHE_TTL = 3600  # 缓存 1 小时

# 动态获取时关注的热门仓库前缀（按下载量排序的 GGUF 模型）
_DYNAMIC_SEARCH_TERMS = [
    "GGUF",
]
_DYNAMIC_LIMIT = 40  # 最多拉取数量


def fetch_dynamic_catalog(force: bool = False) -> dict:
    """从 HuggingFace API 动态获取热门 GGUF 模型。
    返回 {"models": [...], "source": "dynamic", "updated_at": timestamp}
    失败时返回 {"models": [], "error": "..."}
    """
    global _dynamic_cache, _dynamic_cache_time

    # 使用缓存（非强制刷新且未过期）
    if not force and _dynamic_cache and (time.time() - _dynamic_cache_time) < _DYNAMIC_CACHE_TTL:
        return {"models": _dynamic_cache, "source": "dynamic",
                "updated_at": _dynamic_cache_time, "cached": True}

    try:
        # 搜索最近更新的热门 GGUF 模型仓库（镜像站优先，失败回退原始站）
        # 注意：系统 Python 3.9 的 SSL 库可能无法连接某些站点，改用 curl 子进程
        query = (f"?search=GGUF&sort=downloads&direction=-1"
                 f"&limit={_DYNAMIC_LIMIT}&filter=text-generation")
        data = None
        last_err = None
        for api_base in (_HF_API_MIRROR, _HF_API_ORIGIN):
            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "15", api_base + query],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout)
                    break
                else:
                    last_err = Exception(f"curl 返回码 {result.returncode}")
            except Exception as e:
                last_err = e
        if data is None:
            raise last_err or Exception("所有 API 源均不可达")

        models = []
        for item in data:
            repo_id = item.get("id", "")
            if not repo_id:
                continue
            # 只保留 GGUF 仓库
            if "gguf" not in repo_id.lower():
                continue
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)
            updated = item.get("lastModified", "")
            # 提取模型名（去掉 -GGUF 后缀）
            name = repo_id.split("/")[-1].replace("-GGUF", "").replace("-gguf", "")
            models.append({
                "id": f"dyn-{repo_id.replace('/', '-')}",
                "name": name,
                "repo": repo_id,
                "downloads": downloads,
                "likes": likes,
                "updated_at": updated,
                "url": f"{HF_MIRROR_BASE}/{repo_id}",
                "source": "dynamic",
            })

        _dynamic_cache = models
        _dynamic_cache_time = time.time()
        return {"models": models, "source": "dynamic",
                "updated_at": _dynamic_cache_time, "cached": False}

    except Exception as e:
        # 网络失败时返回缓存（如果有）
        if _dynamic_cache:
            return {"models": _dynamic_cache, "source": "dynamic",
                    "updated_at": _dynamic_cache_time, "cached": True,
                    "error": f"刷新失败（{e}），显示缓存数据"}
        return {"models": [], "source": "dynamic", "error": str(e)}


# ==================== 查询接口 ====================

def list_all(source: str = DEFAULT_SOURCE, category: Optional[str] = None) -> List[dict]:
    """列出模型；category=None 全部，'text'/'video' 按类别筛选"""
    return [m.to_dict(source) for m in CATALOG
            if category is None or m.category == category]


def get_by_id(model_id: str):
    for m in CATALOG:
        if m.id == model_id:
            return m
    return None


def filter_by_vram(vram_gb: float, source: str = DEFAULT_SOURCE,
                   category: Optional[str] = None) -> List[dict]:
    """按可用显存筛选：返回显存足够跑的模型；category 可选按文本/视频过滤"""
    return [m.to_dict(source) for m in CATALOG
            if m.min_vram_gb <= vram_gb
            and (category is None or m.category == category)]
