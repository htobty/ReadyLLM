"""视频提示词编排（路线 A：LLM 参数编排 + 官方结构化格式）

把用户一句话的粗略画面描述，交给大模型扩写，再按 MiniMax H3 官方提示词
格式（见 h3_prompt_format.py）拼装成最终 prompt。目标是"用户不自己编排，
也能拿到踩在 H3 训练分布上的高质量出片"。

为什么不让 LLM 直接吐最终 prompt：实测自由英文散文质量差，根因是没踩到
H3 的结构化字段格式（integrated_multimodal_description / retention_analysis
等）。格式正确性交给代码拼装保证，LLM 只负责填语义内容，避免它每次记模板
记飘。

设计要点：
  - 复用 ai_tuner 的 LLM 配置通道（get_config 读 ai_config.json），不引入
    新 API 配置项。
  - 单独实现高 temperature 调用：ai_tuner._call_llm 写死 temp=0.3（参数精
    调求稳），创意扩写需要多样性，故只复用其配置读取与 JSON 容错解析。
  - 失败一律返回 None，由调用方（/generate）优雅降级为原始 prompt。
  - 不硬编码任何个人环境：模型名 / 接口地址全部来自用户已保存的 ai_config。
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from .ai_tuner import get_config, _parse_llm_response
from .h3_prompt_format import compose_prompt


# H3 turbo 工作流的合理参数区间（编排结果 clamp 到此范围，避免 LLM 乱给）
_STEPS_MIN, _STEPS_MAX = 4, 20
_CFG_MIN, _CFG_MAX = 0.5, 8.0


# 通用撰写规则：LLM 只填语义内容，不写最终字段名（字段由代码拼装）
_COMMON_RULES = """撰写规则（决定成片质量，务必遵守）：
1. shot1 用英文（H3 对英文响应最好），40-80 词，自然成句不要堆逗号关键词。
2. shot1 开头先声明风格与景别，如 "Cinematic, live-action, a medium-wide shot frames ..."；
   风格从用户意图或参考图推断（写实 / 2D-animated / 3D CG / watercolor 等）。
3. 运镜写成自然英文动作句，含「运动类型 + 幅度 + 速度」三维度，例如：
   "The camera pushes in with small amplitude at slow speed toward ..."。
   运动类型词表：Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right,
   Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot。
   幅度/速度无意义时可省略（中幅常速默认不写）。
4. 明确光影氛围（soft morning light / golden hour / moody volumetric lighting）。
5. soundscape：1-4 句英文，描述画面里能听见的真实声音（环境音 + 物理动作音 +
   人声），如水声、脚步、风声、布料摩擦。
6. music：1-2 句英文，描述观众独享的背景配乐（乐器 + 速度 + 情绪）；用户没提
   配乐或要安静时填 "N/A"。
7. 保留用户原始意图的所有关键元素，不凭空添加会改变主体的人物或情节。"""

# R2V 专属：身份锚定是质量关键
_R2V_RULES = """

这是「参考图生视频」，必须用官方六段式锁人物身份：
- subjects：为每个主体给稳定 name（如 "Subject 1"）、refs（它出现在哪些参考图，
  如 ["Picture 1","Picture 2"]）、appearance（具体可见特征：脸型/发型/发饰/服装
  颜色款式等，越具体越好，不要写"唯美/电影感"这类空标签）。
- summary：一句话概述整段视频谁在做什么，用 <Subject N> 引用主体。
- retention：对每个主体列 preserved——明确哪些特征必须保持不变（fully_preserved），
  这是 H3 锁身份的核心，务必把 subjects 里的关键外观特征复述进来。
- shot1 里描述动作时也要用 <Subject N> 引用主体，保持身份一致。"""


SYSTEM_PROMPT_T2V = """你是一位资深电影摄影师与分镜师，为 MiniMax H3 文生视频模型撰写画面内容。

用户会给你一句中文或英文的画面描述。输出一个 JSON 对象：
{
  "shot1": "英文：本镜头的风格声明 + 主体 + 动作 + 场景 + 运镜（三维度自然句）+ 光影",
  "soundscape": "英文：画面内真实声音",
  "music": "英文：背景配乐，无则 N/A",
  "steps": 整数(4-20),
  "cfg": 小数(0.5-8.0),
  "reasoning": "中文简述编排思路"
}

""" + _COMMON_RULES + """

参数规则：steps 写实电影质感 8-12、默认 8；cfg H3 turbo 推荐 1.0 附近，描述很具体
希望强跟随用 1.5-2.5，默认 1.0。

只输出 JSON，不要任何额外文字或 markdown 之外的内容。"""


SYSTEM_PROMPT_R2V = """你是一位资深电影摄影师与分镜师，为 MiniMax H3 参考图生视频（Ref2VA）撰写内容。

用户会给你一句画面描述（画面里的人物/主体已由参考图提供）。输出一个 JSON 对象：
{
  "subjects": [{"name": "Subject 1", "refs": ["Picture 1","Picture 2"], "appearance": "英文具体外观特征"}],
  "summary": "英文：一句话概述整段视频谁在做什么，用 <Subject N> 引用",
  "retention": [{"name": "Subject 1", "preserved": "英文：必须保持不变的具体特征"}],
  "shot1": "英文：风格声明 + 用 <Subject N> 描述动作 + 场景 + 运镜（三维度）+ 光影",
  "soundscape": "英文：画面内真实声音",
  "music": "英文：背景配乐，无则 N/A",
  "steps": 整数(4-20),
  "cfg": 小数(0.5-8.0),
  "reasoning": "中文简述编排思路"
}

""" + _COMMON_RULES + _R2V_RULES + """

参数规则：steps 默认 8；cfg 默认 1.0。

只输出 JSON，不要任何额外文字。"""


def _call_llm_creative(cfg: Dict[str, Any], messages: list,
                       temperature: float = 0.85,
                       max_tokens: int = 1024) -> Optional[str]:
    """高 temperature 的 OpenAI 兼容调用（创意扩写用，区别于调优的低温调用）。"""
    url = cfg.get("api_url", "").rstrip("/")
    if not url.endswith("/chat/completions"):
        if "/v1" in url:
            url = f"{url}/chat/completions"
        else:
            url = f"{url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    payload = json.dumps({
        "model": cfg.get("model_name", ""),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return None
    return None


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def enhance_prompt(user_prompt: str, mode: str = "t2v",
                   picture_count: int = 1) -> Optional[Dict[str, Any]]:
    """把用户粗略描述编排成 H3 官方结构化 prompt + 推荐参数。

    mode: t2v / i2v / r2v。r2v 走六段式（含身份锚定），其余走三核心字段。
    picture_count: 参考图/首帧图数量（r2v 用于给 subjects 分配 Picture 标签）。

    成功返回 {"prompt"(已拼装成官方格式), "steps", "cfg", "reasoning"}；
    未配置 LLM / 调用失败 / 解析失败 一律返回 None（调用方降级用原 prompt）。
    """
    if not user_prompt or not user_prompt.strip():
        return None

    cfg = get_config()
    if not cfg.get("api_url") or not cfg.get("model_name"):
        # 用户还没配 LLM，无法编排，交回原样
        return None

    sys_prompt = SYSTEM_PROMPT_R2V if mode == "r2v" else SYSTEM_PROMPT_T2V
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]
    raw = _call_llm_creative(cfg, messages,
                             max_tokens=1600 if mode == "r2v" else 1024)
    parsed = _parse_llm_response(raw)
    if not parsed or not parsed.get("shot1"):
        return None

    # 按官方模板拼装成最终 prompt（格式正确性由代码保证）
    final = compose_prompt(parsed, mode=mode, picture_count=picture_count)
    if not final.strip():
        return None

    steps = int(_clamp(parsed.get("steps"), _STEPS_MIN, _STEPS_MAX, 8))
    cfd = _clamp(parsed.get("cfg"), _CFG_MIN, _CFG_MAX, 1.0)
    return {
        "prompt": final,
        "steps": steps,
        "cfg": round(cfd, 2),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }
