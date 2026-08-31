"""H3 官方提示词格式拼装器（单一事实来源）。

依据 MiniMax-H3 官方 skill：.agents/skills/h3-prompt-writing/
  - references/base-en.txt（T2VA / I2VA：三核心字段）
  - references/ref-en.txt（Ref2VA：六段式）

设计原则：LLM 只负责填「语义内容」（主体外观、动作、运镜、声音等），
本模块负责把这些内容按官方固定字段名、固定顺序、固定句式拼装成最终
prompt 字符串。格式正确性由代码保证，不依赖 LLM 每次记对模板。

三种模式：
  t2v  纯文生视频：integrated_multimodal_description + overall_soundscape + non_diegetic_music
  i2v  图生视频：上面三字段 + 首行 I2VA 对齐指令（固定句式）
  r2v  参考图生视频：六段式 subject_definitions/summary/retention_analysis/
       detailed_description/overall_soundscape/non_diegetic_music

我们每段视频是独立生成的短视频，detailed_description 内只放单个 [Shot 1]
（首镜头不加时间戳，符合官方「first shot 不加 timestamp」规则）。多镜头时间戳
留给未来单段内多 shot 的场景。
"""

from typing import Dict, Any, List


def _shot_body(shot1: str) -> str:
    """把 shot 内容规整为 [Shot 1] 开头声明风格的正文。"""
    s = (shot1 or "").strip()
    if s.lower().startswith("[shot 1]"):
        return s
    return "[Shot 1] " + s


def _ref_phrase(refs: List[str]) -> str:
    """['Picture 1','Picture 2'] -> '<Picture 1> and <Picture 2>'。"""
    tags = ["<%s>" % r.strip() for r in refs if r and r.strip()]
    if not tags:
        return ""
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]


def compose_ref_prompt(data: Dict[str, Any]) -> str:
    """R2V（Ref2VA）六段式。data 需含 subjects/summary/retention/shot1/
    soundscape/music。缺字段则跳过该段，保证不崩。"""
    subjects = data.get("subjects") or []
    retention = data.get("retention") or []
    parts: List[str] = []

    # 1. subject_definitions
    if subjects:
        lines = []
        for sub in subjects:
            name = str(sub.get("name", "")).strip()
            if not name:
                continue
            rp = _ref_phrase(sub.get("refs") or [])
            app = str(sub.get("appearance", "")).strip()
            lead = "<%s>" % name
            if rp:
                lines.append("%s is %s, %s." % (lead, rp, app) if app
                             else "%s is %s." % (lead, rp))
            else:
                lines.append("%s is %s." % (lead, app))
        if lines:
            parts.append("subject_definitions:\n" + "\n".join(lines))

    # 2. summary
    summary = str(data.get("summary", "")).strip()
    if summary:
        if not summary.startswith("["):
            summary = "[reference generation] " + summary
        parts.append("summary:\n" + summary)

    # 3. retention_analysis
    if retention:
        rlines = []
        for r in retention:
            name = str(r.get("name", "")).strip()
            if not name:
                continue
            preserved = str(r.get("preserved", "")).strip()
            # LLM 有时把 "fully_preserved" 写进 preserved 字段本身，与下面模板
            # 的 "fully_preserved - " 前缀重复，剥掉它保证只出现一次。
            if preserved.lower().startswith("fully_preserved"):
                preserved = preserved[len("fully_preserved"):].lstrip(" -–:：").strip()
            rlines.append("<%s> (appears in [Shot 1]): fully_preserved - %s"
                          % (name, preserved))
        if rlines:
            parts.append("retention_analysis:\n" + "\n".join(rlines))

    # 4. detailed_description
    shot1 = str(data.get("shot1", "")).strip()
    if shot1:
        parts.append("detailed_description:\n" + _shot_body(shot1))

    # 5. overall_soundscape
    scape = str(data.get("soundscape", "")).strip()
    if scape:
        parts.append("overall_soundscape:\n" + scape)

    # 6. non_diegetic_music
    music = str(data.get("music", "")).strip() or "N/A"
    parts.append("non_diegetic_music:\n" + music)

    return "\n\n".join(parts)


def compose_base_prompt(data: Dict[str, Any], mode: str = "t2v",
                        picture_count: int = 1) -> str:
    """T2VA / I2VA 三核心字段。mode=i2v 时首行加固定对齐指令。"""
    parts: List[str] = []

    # I2VA 对齐指令行（官方固定句式）
    if mode == "i2v":
        parts.append(
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.")

    shot1 = str(data.get("shot1", "")).strip()
    if shot1:
        parts.append("integrated_multimodal_description: " + _shot_body(shot1))

    scape = str(data.get("soundscape", "")).strip()
    if scape:
        parts.append("overall_soundscape: " + scape)

    music = str(data.get("music", "")).strip() or "N/A"
    parts.append("non_diegetic_music: " + music)

    return "\n\n".join(parts)


def compose_prompt(data: Dict[str, Any], mode: str = "t2v",
                   picture_count: int = 1) -> str:
    """按模式分派。mode ∈ {t2v, i2v, r2v}。"""
    if mode == "r2v":
        return compose_ref_prompt(data)
    return compose_base_prompt(data, mode=mode, picture_count=picture_count)


if __name__ == "__main__":
    # 自测：拼装输出应符合官方六段式结构
    d = {
        "subjects": [{"name": "Subject 1",
                      "refs": ["Picture 1", "Picture 2"],
                      "appearance": "the young woman with an oval face, "
                                    "black hanging bun with a white jade hairpin, "
                                    "pale-cyan qixiong ruqun Hanfu"}],
        "summary": "The target video shows <Subject 1> washing cloth by a canal.",
        "retention": [{"name": "Subject 1",
                       "preserved": "the oval face, hanging bun with jade hairpin, "
                                    "and pale-cyan Hanfu are retained."}],
        "shot1": "Cinematic, live-action, a medium side-view shot frames "
                 "<Subject 1> crouching beside blue stone steps. The camera "
                 "pushes in with small amplitude at slow speed.",
        "soundscape": "Gentle lapping of canal water and soft fabric rustling.",
        "music": "A restrained solo guqin melody at a slow tempo.",
    }
    print("===== R2V 六段式 =====")
    print(compose_ref_prompt(d))
    print("\n===== I2V 三字段 =====")
    print(compose_base_prompt({"shot1": d["shot1"],
                               "soundscape": d["soundscape"],
                               "music": d["music"]}, mode="i2v"))
