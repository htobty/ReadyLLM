"""分镜脚本生成器（长视频第 1 层）+ H3 官方结构化格式

把一个主题/故事交给大模型拆成「多镜头分镜脚本」。每个镜头（shot）对应一段可被
H3 单次生成的短视频，最终由 video_pipeline 串行 R2V 生成并拼接成分钟级长视频。

关键：每个 shot 的 prompt 字段直接是 H3 官方 Ref2VA 六段式（subject_definitions/
summary/retention_analysis/detailed_description/overall_soundscape/non_diegetic_music），
由 h3_prompt_format.compose_ref_prompt 拼装。实测官方格式比自由散文质量明显改善，
且 video_pipeline._generate_one_shot 直接喂 shot["prompt"]，无需改动 pipeline。

设计要点：
  - 全局 subjects/retention 一次定义、所有段复用（同一组参考图锁身份），每段只
    换 detailed_description(shot1)/soundscape/music。
  - 复用 video_prompt._call_llm_creative（高 temperature 创意通道）与 ai_tuner 配置。
  - 每段同时给 first_frame_prompt（喂 MakWo 文生图出首帧/参考图）。
  - 失败返回 None，调用方决定降级。
  - 不硬编码任何个人环境。
"""

import json
from typing import Optional, Dict, Any, List

from .ai_tuner import _parse_llm_response
from .video_prompt import _call_llm_creative, get_config
from .h3_prompt_format import compose_ref_prompt


# 单段合理帧数区间（@16fps：80 帧≈5s，161 帧≈10s），兼顾连贯性与显存
_LEN_MIN, _LEN_MAX = 49, 161


def _align_len(n: int) -> int:
    n = max(_LEN_MIN, min(_LEN_MAX, int(n)))
    k = round((n - 1) / 17.0)
    aligned = 17 * k + 1
    if aligned < _LEN_MIN:
        aligned = _LEN_MIN
    if aligned > _LEN_MAX:
        aligned = _LEN_MAX
    return aligned


SYSTEM_PROMPT = """你是一位电影导演兼分镜师。用户给你一个视频主题（可能含故事梗概）和期望总时长，你要把它拆成一串连续镜头，最终由 MiniMax H3 参考图生视频（Ref2VA）逐段生成再拼接。

输出一个 JSON 对象：
{
  "title": "视频标题",
  "subjects": [{"name": "Subject 1", "refs": ["Picture 1","Picture 2"], "appearance": "英文：主体具体可见特征（脸型/发型/发饰/服装颜色款式等，越具体越好）"}],
  "retention": [{"name": "Subject 1", "preserved": "英文：必须保持不变的关键外观特征"}],
  "shots": [
    {
      "index": 1,
      "title": "该镜头简短中文名",
      "shot1": "英文：本镜头[Shot 1]正文——风格声明 + 用 <Subject N> 描述动作 + 场景 + 运镜 + 光影（40-70词）",
      "soundscape": "英文：本镜头画面内真实声音",
      "music": "英文：背景配乐，无则 N/A",
      "first_frame_prompt": "英文：本镜头第一帧静态画面提示词，用于文生图（构图/主体/光线，30-50词）",
      "length": 整数(49-161，16fps 下帧数≈秒数×16)
    }
  ]
}

拆解规则（决定成片质量，务必遵守）：
1. 镜头数 = 期望总时长 ÷ 单段约 6-8 秒，向上取整，通常 4-12 个。
2. 每段 length 落在 49-161（约 3-10 秒），各段之和≈期望总时长×16。
3. subjects 与 retention 是全局的：定义一次，所有镜头复用同一组（这是 H3 锁人物身份的核心，retention.preserved 必须复述 subjects 里的关键外观特征）。
4. 每个 shot1 用 <Subject N> 引用主体描述动作；镜头之间要有叙事推进或视觉变化，但主体/场景/色调保持一致。
5. 运镜写成自然英文动作句，含运动类型+幅度+速度三维度，如 "The camera pushes in with small amplitude at slow speed toward ..."。运动类型词表：Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Arc Shot, Tracking Shot, Static Shot。
6. shot1 开头声明风格与景别，如 "Cinematic, live-action, a medium shot frames ..."。
7. first_frame_prompt 必须能独立生成高质量静帧，且与 subjects 的外貌描述一致。

只输出 JSON，不要任何额外文字。"""


def generate_storyboard(theme: str, total_seconds: int = 60,
                        max_shots: int = 12) -> Optional[Dict[str, Any]]:
    """把主题拆成分镜脚本，每段 prompt 为 H3 官方六段式。

    成功返回 {"title", "shots": [{"index","title","prompt"(官方六段式),
    "first_frame_prompt","length"}, ...]}；未配 LLM/失败返回 None。
    """
    if not theme or not theme.strip():
        return None
    cfg = get_config()
    if not cfg.get("api_url") or not cfg.get("model_name"):
        return None

    user_msg = (
        f"主题：{theme.strip()}\n"
        f"期望总时长：约 {int(total_seconds)} 秒\n"
        f"最多 {max_shots} 个镜头。请输出分镜脚本 JSON。"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = _call_llm_creative(cfg, messages, temperature=0.8, max_tokens=4000)
    parsed = _parse_llm_response(raw)
    if not parsed or not isinstance(parsed.get("shots"), list) or not parsed["shots"]:
        return None

    # 全局身份定义（所有段复用）
    subjects = parsed.get("subjects") or []
    retention = parsed.get("retention") or []

    shots: List[Dict[str, Any]] = []
    for i, s in enumerate(parsed["shots"][:max_shots]):
        if not s.get("shot1"):
            continue
        # 用拼装器把本段语义内容组装成官方六段式（格式正确性由代码保证）
        data = {
            "subjects": subjects,
            "summary": s.get("summary") or (
                "The target video shows " + (subjects[0]["name"] if subjects else "the subject")
                + " in this shot."),
            "retention": retention,
            "shot1": s.get("shot1", ""),
            "soundscape": s.get("soundscape", ""),
            "music": s.get("music", ""),
        }
        prompt = compose_ref_prompt(data)
        if not prompt.strip():
            continue
        shots.append({
            "index": i + 1,
            "title": str(s.get("title", f"镜头{i+1}")).strip(),
            "prompt": prompt,
            "first_frame_prompt": str(
                s.get("first_frame_prompt", s.get("shot1", ""))).strip(),
            "length": _align_len(s.get("length", 97)),
        })
    if not shots:
        return None
    return {"title": str(parsed.get("title", theme)).strip(), "shots": shots}
