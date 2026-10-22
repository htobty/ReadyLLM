"""文生图提示词编排。

视频那条链路（video_prompt.py）产出的是 MiniMax H3 的结构化分镜 prompt，
直接喂给 SDXL / Qwen-Image 会水土不服（它们吃的是画面描述＋质量标签，
不是镜头/动作/音效字段）。所以图片单独一套 system prompt，只做一件事：
把用户一句口语描述扩写成模型能吃好的画面描述，顺带给出保守的步数建议。

拿不到 LLM（未配置 / 调用失败 / 解析失败）时返回 None，调用方降级用原
prompt，绝不阻断出图。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

SYSTEM_PROMPT_IMAGE = """你是一位文生图提示词工程师，服务对象是 Stable Diffusion /
SDXL / Qwen-Image 这类模型。用户会给一句很口语的画面描述，你要把它扩写成
模型容易出好图的提示词。

要求：
1. 用英文输出 prompt（这些模型对英文标签更敏感），主体 + 场景 + 构图 +
   光线 + 质感，5 句以内，不要堆砌无意义的形容词。
2. 不要写镜头运动、时间轴、音效这类视频才有的东西。
3. 不要写分辨率、采样器、步数等参数词，参数由调用方控制。
4. 保持用户原意，不要凭空加入人物身份、品牌、文字。

只输出 JSON，不要任何解释或 Markdown 代码块：
{"prompt": "英文提示词", "reasoning": "一句中文说明你补了哪些关键信息"}"""


def _get_config() -> Dict[str, Any]:
    """复用视频链路已经在用的 LLM 配置，避免两套配置入口。"""
    try:
        from .video_prompt import get_config  # noqa: WPS433 (延迟导入，避免循环)
        return get_config() or {}
    except Exception:
        return {}


def _endpoint(api_url: str) -> str:
    """把配置里的 api_url 归一成 chat/completions 完整地址。"""
    u = (api_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions"):
        return u
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u + "/chat/completions"


def _call_llm(cfg: Dict[str, Any], user_prompt: str) -> Optional[str]:
    url = _endpoint(cfg.get("api_url") or "")
    model = cfg.get("model_name") or ""
    if not url or not model:
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_IMAGE},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 600,
    }
    headers = {"Content-Type": "application/json"}
    key = cfg.get("api_key") or ""
    if key:
        headers["Authorization"] = "Bearer " + key

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _parse(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return None
    prompt = (obj.get("prompt") or "").strip()
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "reasoning": (obj.get("reasoning") or "").strip(),
    }


def enhance_prompt(user_prompt: str) -> Optional[Dict[str, Any]]:
    """把口语描述扩写成文生图提示词；失败一律返回 None 由调用方降级。"""
    if not user_prompt or not user_prompt.strip():
        return None
    cfg = _get_config()
    raw = _call_llm(cfg, user_prompt.strip())
    if not raw:
        return None
    return _parse(raw)
