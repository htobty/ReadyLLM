"""端到端验证核心：LLM 自动编排 → 拼装器输出，校验是否合格官方六段式。
不依赖 ComfyUI，直接调 enhance_prompt(mode=r2v)，检查返回的 final_prompt。"""
import sys
sys.path.insert(0, ".")
from app.services.video_prompt import enhance_prompt

PROMPT = ("古风汉服少女在江南风格的小河边蹲着洗衣服，清晨薄雾，"
          "镜头缓缓推近，氛围安静唯美")

r = enhance_prompt(PROMPT, mode="r2v", picture_count=2)
if not r:
    print("FAIL: enhance_prompt 返回 None（LLM 未配置/调用失败/解析失败）")
    sys.exit(1)

fp = r["prompt"]
print("===== 自动拼装的 final_prompt =====")
print(fp)
print("\n===== steps=%s cfg=%s =====" % (r["steps"], r["cfg"]))
print("reasoning:", r["reasoning"])

# 格式校验：六段式必备字段
checks = {
    "subject_definitions:": "subject_definitions:" in fp,
    "summary:": "summary:" in fp,
    "retention_analysis:": "retention_analysis:" in fp,
    "detailed_description:": "detailed_description:" in fp,
    "overall_soundscape:": "overall_soundscape:" in fp,
    "non_diegetic_music:": "non_diegetic_music:" in fp,
    "[Shot 1]": "[Shot 1]" in fp,
    "fully_preserved": "fully_preserved" in fp,
    "<Subject": "<Subject" in fp,
    "<Picture": "<Picture" in fp,
    "运镜三维度(amplitude/speed)": ("amplitude" in fp or "speed" in fp
                              or "Static Shot" in fp),
}
print("\n===== 格式校验 =====")
allok = True
for k, v in checks.items():
    print("  [%s] %s" % ("OK" if v else "缺失", k))
    allok = allok and v
print("\n结论:", "✅ 合格官方六段式" if allok else "⚠️ 有字段缺失，见上")
