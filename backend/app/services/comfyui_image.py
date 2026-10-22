"""文生图：UNETLoader 路线（Qwen-Image 等没有内置 CLIP/VAE 的图片权重）。

为什么要单独一条链路：ComfyUI 的图片模型有两个互不相通的存放位置——

* `checkpoints/`：SD/SDXL 那种把 UNET+CLIP+VAE 打在一个文件里的主权重，
  用 CheckpointLoaderSimple 一步加载；
* `diffusion_models/`：只有一个 UNET（Qwen-Image 2.1 就是这种），得自己
  配 CLIPLoader + VAELoader，而且这个目录是和 MiniMax H3 等视频权重共用的。

之前 /image-models 只扫了 checkpoints，所以用户把 qwen_image 放进
diffusion_models 之后，下拉框里只剩一个 RealVisXL，看着像"图片模型丢了"。
这里补上第二条链路。

CLIP/VAE 的配对没有 API 可查（ComfyUI 不知道哪三个文件是一套），只能按
文件名约定识别。规则写在 _PAIR_RULES 里，只认得出的家族才放行；认不出的
权重（主要是视频 unet）直接不给前端选，避免提交后报缺节点。
"""

from __future__ import annotations

from typing import Optional

# (unet 名关键字, CLIP 名关键字优先级, VAE 名关键字优先级)
# 注意 CLIP 的关键字顺序有讲究：本机同时存在 qwen3vl_32b（H3 的文本编码器）
# 和 qwen3vl_8b（Qwen-Image 的），两者都含 "qwen"，所以必须用更长的
# "qwen3vl_8b" 先命中，否则会把视频的 CLIP 配给图片模型。
_PAIR_RULES = [
    ("qwen_image", ("qwen3vl_8b", "qwen3vl", "qwen"), ("qwen_image", "qwen")),
    ("qwen-image", ("qwen3vl_8b", "qwen3vl", "qwen"), ("qwen_image", "qwen")),
]

# Qwen-Image 的官方采样参数：euler + simple，20 步，cfg 2.5，shift 3.1。
# cfg 给到 SDXL 那种 6~7 会过曝并出现硬边，所以这条链路的默认值单独一套。
QWEN_IMAGE_DEFAULTS = {
    "steps": 20,
    "cfg": 2.5,
    "sampler": "euler",
    "scheduler": "simple",
    "shift": 3.1,
}


def _options(engine, node: str, field: str) -> list:
    """从 /object_info/<node> 取某个输入的可选值列表。

    走 engine 自己的 _curl_json，和 list_checkpoints 用的是同一个通道
    （SSH 隧道 / 直连都由引擎决定），这里不另建 HTTP 客户端。
    """
    try:
        info = engine._curl_json(f"/object_info/{node}", timeout=15)
    except Exception:
        return []
    if not info:
        return []
    try:
        opts = info[node]["input"]["required"][field][0]
    except (KeyError, IndexError, TypeError):
        return []
    return [o for o in opts if isinstance(o, str)]


def list_unets(engine) -> list:
    """diffusion_models 目录里的全部 unet 权重（图片和视频混在一起）。"""
    return _options(engine, "UNETLoader", "unet_name")


def list_clips(engine) -> list:
    return _options(engine, "CLIPLoader", "clip_name")


def list_vaes(engine) -> list:
    return _options(engine, "VAELoader", "vae_name")


def _pick(names: list, keywords: tuple) -> str:
    """按关键字优先级挑第一个命中的文件名；都不命中返回空串。"""
    low = [(n, n.lower()) for n in names]
    for kw in keywords:
        for n, ln in low:
            if kw in ln:
                return n
    return ""


def pair_for(unet_name: str, clips: list, vaes: list) -> Optional[tuple]:
    """给图片 unet 配一套 (clip_name, vae_name)；认不出的家族返回 None。"""
    if not unet_name:
        return None
    low = unet_name.lower()
    for unet_kw, clip_kws, vae_kws in _PAIR_RULES:
        if unet_kw not in low:
            continue
        clip = _pick(clips, clip_kws)
        # 视频 VAE（minimax_h3_*）名字里也带 vae，所以 VAE 优先按家族名匹配，
        # 匹配不到再退回泛化关键字，并在泛化时排除明显属于视频的。
        vae = _pick(vaes, vae_kws)
        if not vae:
            vae = _pick([v for v in vaes if "minimax" not in v.lower()], ("vae",))
        if clip and vae:
            return clip, vae
        return None
    return None


def is_image_unet(unet_name: str) -> bool:
    low = (unet_name or "").lower()
    return any(kw in low for kw, _, _ in _PAIR_RULES)


def resolve_name(name: str, options: list) -> str:
    """把前端传来的模型名归一到 ComfyUI 列表里的真实文件名。

    ComfyUI 的 UNETLoader 等节点做的是「值必须在列表里」的严格校验，只认
    ``qwen_image_2.1_int8_convrot.safetensors`` 这种完整文件名；而前端下拉框
    显示的、以及 localStorage 里缓存过的都可能是去掉扩展名的显示名。少了
    扩展名，整张图会被 prompt_outputs_failed_validation 直接拒掉，报错里还
    只有一句 "Value not in list"。

    注意不能对传入名 rsplit('.')：Qwen-Image 这种名字里自带版本号点号
    （``qwen_image_2.1_vae_bf16``），按最后一个点切会把 2.1 也切成扩展名。
    正确做法是只剥列表侧已知的权重扩展名，再和传入名整体比较。
    """
    if not name or not options:
        return name
    if name in options:
        return name
    # 列表侧去扩展名后与传入名整体相等（前端只给了显示名）
    hits = [o for o in options if o.rsplit(".", 1)[0] == name]
    if hits:
        return hits[0]
    # 传入名自带扩展名，但与列表侧的扩展名不同（.ckpt 对 .safetensors）
    for ext in _WEIGHT_EXTS:
        if name.lower().endswith(ext):
            bare = name[: -len(ext)]
            hits = [o for o in options if o.rsplit(".", 1)[0] == bare]
            return hits[0] if hits else name
    return name


def build_unet_image_workflow(
    prompt: str,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = QWEN_IMAGE_DEFAULTS["steps"],
    cfg: float = QWEN_IMAGE_DEFAULTS["cfg"],
    seed: Optional[int] = None,
    sampler: str = QWEN_IMAGE_DEFAULTS["sampler"],
    scheduler: str = QWEN_IMAGE_DEFAULTS["scheduler"],
    shift: float = QWEN_IMAGE_DEFAULTS["shift"],
    filename_prefix: str = "modeldeploy/image",
) -> dict:
    """Qwen-Image 一类的 unet 文生图 workflow。

    节点编号从 400 起，避开视频链路（1xx）和 checkpoint 链路（3xx），
    方便在 ComfyUI 的报错里一眼认出是哪条路走的。
    EmptySD3LatentImage 而不是 EmptyLatentImage：前者是 16 通道，
    匹配 Qwen-Image / SD3 这类 DiT 的 latent 形状。
    """
    if seed is None:
        seed = 0
    return {
        "400": {"class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "401": {"class_type": "CLIPLoader",
                "inputs": {"clip_name": clip_name, "type": "qwen_image"}},
        "402": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "403": {"class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["401", 0]}},
        "404": {"class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt or "", "clip": ["401", 0]}},
        "405": {"class_type": "EmptySD3LatentImage",
                "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "406": {"class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["400", 0], "shift": float(shift)}},
        "407": {"class_type": "KSampler",
                "inputs": {"model": ["406", 0], "positive": ["403", 0],
                           "negative": ["404", 0], "latent_image": ["405", 0],
                           "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                           "sampler_name": sampler, "scheduler": scheduler,
                           "denoise": 1.0}},
        "408": {"class_type": "VAEDecode",
                "inputs": {"samples": ["407", 0], "vae": ["402", 0]}},
        "409": {"class_type": "SaveImage",
                "inputs": {"images": ["408", 0], "filename_prefix": filename_prefix}},
    }
