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



# ---------------------------------------------------------------------------
# 图片模型精选清单
#
# 图片模型和文本/视频模型在磁盘布局上不一样：文本模型是单个 GGUF 文件，
# 视频模型是单个 safetensors，而图片模型通常拆成 unet/clip/vae 三段。
# 所以这里单独维护一份 dict 清单，字段与前端 ModelCard 对齐，并额外给出
# loader 信息（checkpoint / unet），下载时按 loader 决定落到哪个目录。
# ---------------------------------------------------------------------------
IMAGE_MODELS: List[dict] = [
    {
        "id": "img-qwen-image",
        "name": "Qwen-Image",
        "quant": "BF16",
        "desc": "阿里通义千问的图像生成模型，中英双语文字渲染强，适合海报、带字图、写实照片。",
        "size_gb": 41.0,
        "min_vram_gb": 20.0,
        "filename": "qwen_image_2.1_int8_convrot.safetensors",
        "loader": "unet",
        "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
        "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
        "category": "image",
        "engine": "comfyui",
        "repo": "Qwen/Qwen-Image",
        "tags": ["image", "qwen", "t2i", "text-render"],
        "available_sources": ["hf-mirror", "modelscope", "huggingface"],
        "steps": 20,
        "cfg": 2.5,
    },
    {
        "id": "img-flux-dev",
        "name": "FLUX.1-dev",
        "quant": "BF16",
        "desc": "Black Forest Labs 的 12B 整流流模型，构图与提示词跟随最好，适合概念图、写实人像。",
        "size_gb": 23.8,
        "min_vram_gb": 16.0,
        "filename": "flux1-dev.safetensors",
        "loader": "unet",
        "clip_name": "t5xxl_fp16.safetensors",
        "vae_name": "ae.safetensors",
        "category": "image",
        "engine": "comfyui",
        "repo": "black-forest-labs/FLUX.1-dev",
        "tags": ["image", "flux", "t2i"],
        "available_sources": ["hf-mirror", "huggingface"],
        "steps": 25,
        "cfg": 3.5,
    },
    {
        "id": "img-flux-schnell",
        "name": "FLUX.1-schnell",
        "quant": "BF16",
        "desc": "FLUX 的蒸馏版，4 步出图，速度是 dev 的五六倍，适合批量出草稿。",
        "size_gb": 23.8,
        "min_vram_gb": 12.0,
        "filename": "flux1-schnell.safetensors",
        "loader": "unet",
        "clip_name": "t5xxl_fp16.safetensors",
        "vae_name": "ae.safetensors",
        "category": "image",
        "engine": "comfyui",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "tags": ["image", "flux", "fast", "t2i"],
        "available_sources": ["hf-mirror", "huggingface"],
        "steps": 4,
        "cfg": 1.0,
    },
    {
        "id": "img-realvisxl-v4",
        "name": "RealVisXL V4.0",
        "quant": "BF16",
        "desc": "基于 SDXL 的写实向 checkpoint，人像与皮肤质感好，单文件加载最快。",
        "size_gb": 6.5,
        "min_vram_gb": 8.0,
        "filename": "RealVisXL_V4.0.safetensors",
        "loader": "checkpoint",
        "category": "image",
        "engine": "comfyui",
        "repo": "SG161222/RealVisXL_V4.0",
        "tags": ["image", "sdxl", "realistic", "checkpoint"],
        "available_sources": ["hf-mirror", "huggingface"],
        "steps": 30,
        "cfg": 5.0,
    },
    {
        "id": "img-sdxl-base",
        "name": "SDXL Base 1.0",
        "quant": "BF16",
        "desc": "Stability 的 SDXL 基座，生态里 LoRA 与控制网最多，8G 显存就能跑。",
        "size_gb": 6.9,
        "min_vram_gb": 8.0,
        "filename": "sd_xl_base_1.0.safetensors",
        "loader": "checkpoint",
        "category": "image",
        "engine": "comfyui",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "tags": ["image", "sdxl", "base", "checkpoint"],
        "available_sources": ["hf-mirror", "huggingface"],
        "steps": 30,
        "cfg": 7.0,
    },
    {
        "id": "img-sd35-large",
        "name": "SD 3.5 Large",
        "quant": "BF16",
        "desc": "Stability 新一代 MMDiT，8B 参数，文字排版和复杂提示词比 SDXL 稳。",
        "size_gb": 16.0,
        "min_vram_gb": 16.0,
        "filename": "sd3.5_large.safetensors",
        "loader": "unet",
        "clip_name": "clip_l.safetensors",
        "vae_name": "sd3.5_vae.safetensors",
        "category": "image",
        "engine": "comfyui",
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "tags": ["image", "sd3.5", "t2i"],
        "available_sources": ["hf-mirror", "huggingface"],
        "steps": 28,
        "cfg": 4.5,
    },
]


# 图片模型的补充规格。
# 条目里的工程字段（filename / loader / steps / cfg）是给下载和 ComfyUI 建图用的，
# 光靠 name 一眼看不出"这是哪个版本、多大、能出多大的图"，所以版本号、参数规模、
# 推荐出图分辨率、真实权重精度单独列一份，由 list_images 合并进条目。
# 精度以 filename 的实际后缀为准：qwen_image_2.1_int8_convrot 是 INT8，不是 BF16。
IMAGE_META: dict = {
    "img-qwen-image": {
        "version": "2.1",
        "params": "20B",
        "precision": "INT8 conv-rot",
        "resolution": "1024×1024，支持 1328×1328 等多比例",
    },
    "img-flux-dev": {
        "version": "1.0-dev",
        "params": "12B",
        "precision": "BF16",
        "resolution": "1024×1024，支持多比例",
    },
    "img-flux-schnell": {
        "version": "1.0-schnell",
        "params": "12B",
        "precision": "BF16",
        "resolution": "1024×1024，支持多比例",
    },
    "img-realvisxl-v4": {
        "version": "V4.0",
        "params": "3.5B（SDXL 架构）",
        "precision": "BF16",
        "resolution": "1024×1024",
    },
    "img-sdxl-base": {
        "version": "1.0",
        "params": "3.5B",
        "precision": "BF16",
        "resolution": "1024×1024",
    },
    "img-sd35-large": {
        "version": "3.5 Large",
        "params": "8B",
        "precision": "BF16",
        "resolution": "1024×1024，支持多比例",
    },
}

# 模型描述英文版（id -> desc_en）。
# 条目的 desc 是中文、写死在目录里；界面默认语言是英文，只给中文描述就会出现
# 「英文界面里突然冒出一句中文」。这里按 id 补英文，前端按当前语言二选一。
# 没列到的条目（如 HF 动态拉取的模型）前端会回退用 desc。
DESC_EN: dict = {
    # --- 文本 ---
    "qwen38-27b": "Newest flagship. Native MTP speculative decoding, smooth on 24 GB.",
    "qwen38-27b-q4km": "Flagship at the standard quant. Best compatibility.",
    "qwen38-8b": "Small flagship with MTP speedup. Runs on 8 GB.",
    "qwen35-32b": "Strong reasoning. Needs 24 GB.",
    "qwen35-14b": "Balanced between quality and speed.",
    "qwen35-8b": "Best value in the lineup.",
    "qwen35-4b": "Light and efficient. Runs on CPU too.",
    "llama4-scout-17b": "MoE with 17B active parameters. Multimodal.",
    "llama4-maverick-17b": "MoE flagship, 128 experts. Needs a large GPU.",
    "deepseek-r1-distill-32b": "Reasoning-tuned. Strong at math and code.",
    "deepseek-r1-distill-14b": "A lighter reasoning model.",
    "deepseek-r1-distill-8b": "Entry-level reasoning model.",
    "gemma3-27b": "Google's multimodal model: vision plus text.",
    "gemma3-12b": "Multimodal at a balanced size.",
    "gemma3-4b": "Lightweight multimodal.",
    "mistral-small-3.2": "Europe's strongest open release. Multilingual.",
    "qwen3-coder-32b": "Strongest open coding model.",
    "qwen3-coder-8b": "Lightweight coding model.",
    "bge-m3": "Multilingual embedding model. A must for RAG.",
    # --- 视频 ---
    "wan21-t2v-1.3b": "Light text-to-video. Runs on 8 GB, 480p and up.",
    "wan21-t2v-14b": "High-quality text-to-video flagship. Needs 24 GB or more.",
    "ltx-video-2b": "Very fast generation on 6 GB. Good for previews.",
    "cogvideox-5b": "Zhipu's open video model, 1280x720. Needs 16 GB.",
    # --- 图片 ---
    "img-qwen-image": "Alibaba's image model. Renders Chinese and English text cleanly, so it suits posters, images with captions and realistic photos.",
    "img-flux-dev": "Black Forest Labs' 12B rectified-flow model. Best prompt following and composition; good for concept art and realistic portraits.",
    "img-flux-schnell": "Distilled FLUX. Four steps per image, several times faster than dev; good for batches of drafts.",
    "img-realvisxl-v4": "SDXL-based checkpoint tuned for realism. Convincing skin and portraits, and the fastest to load as a single file.",
    "img-sdxl-base": "Stability's SDXL base. Largest ecosystem of LoRAs and ControlNets; runs on 8 GB.",
    "img-sd35-large": "Stability's newer MMDiT, 8B parameters. Handles typography and complex prompts better than SDXL.",
}

# 模型商店的分类（文本 / 图片 / 视频），前端筛选栏按这个顺序渲染
STORE_CATEGORIES: List[dict] = [
    {"key": "all", "label_en": "All", "label_zh": "全部"},
    {"key": "text", "label_en": "Text", "label_zh": "文本"},
    {"key": "image", "label_en": "Image", "label_zh": "图片"},
    {"key": "video", "label_en": "Video", "label_zh": "视频"},
]


def list_images(source: str = DEFAULT_SOURCE) -> List[dict]:
    """列出图片模型（按下载源过滤可用性）。

    额外把 IMAGE_META 里的版本/参数量/精度/出图分辨率合并进条目，
    并让 quant 与真实权重精度保持一致（避免出现标 BF16、文件却是 INT8 的情况）。
    """
    out = []
    for m in IMAGE_MODELS:
        item = dict(m)
        item.update(IMAGE_META.get(m["id"], {}))
        if item.get("precision"):
            item["quant"] = item["precision"]
        item["available_sources"] = list(m.get("available_sources") or [])
        item["source_ok"] = source in item["available_sources"]
        item["desc_en"] = DESC_EN.get(m["id"], "")
        out.append(item)
    return out


def get_image_by_id(model_id: str) -> Optional[dict]:
    """按 id 取图片模型条目（商店下载接口用）。"""
    for m in IMAGE_MODELS:
        if m["id"] == model_id:
            return dict(m)
    return None


def list_for_store(source: str = DEFAULT_SOURCE, category: Optional[str] = None) -> List[dict]:
    """商店统一入口：把文本、图片、视频三类的条目合成一份，带 category 字段。

    category 传 None 或 "all" 时返回全部；传 "text"/"image"/"video" 时只返回该类。
    """
    cat = None if category in (None, "", "all") else category
    items: List[dict] = []

    # 直接遍历 CATALOG 取对象，类别以 ModelEntry.category 为准：
    # to_dict() 不一定带 category，setdefault 会把视频模型误标成 text。
    for entry in CATALOG:
        item = entry.to_dict(source)
        item["category"] = getattr(entry, "category", "text") or "text"
        item["source_ok"] = source in (item.get("available_sources") or [])
        item["desc_en"] = DESC_EN.get(entry.id, "")
        items.append(item)

    if cat in (None, "image"):
        items.extend(list_images(source))

    if cat is None:
        # 视频模型已由 list_all 从 CURATED 中带出（其 category="video"）
        pass

    if cat is None or cat == "all":
        return items
    return [m for m in items if m.get("category") == cat]
