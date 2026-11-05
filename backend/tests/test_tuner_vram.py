"""tuner 显存预检的离线单测。

只测纯函数（_Vram / _calibrate / _fits_vram / _score / i18n 键完整性），
不连目标机、不启服务，可离线跑：

    cd backend && PYTHONPATH=. python3 -m unittest discover -s tests -v
"""
import re
import unittest
from pathlib import Path

from app.services.i18n import L, norm_lang, set_lang
from app.services.tuner import _Vram, _calibrate, _fits_vram, _score

GB = 1024 ** 3
TUNER_SRC = Path(__file__).resolve().parents[1] / "app" / "services" / "tuner.py"


def _cfg(ngl="all", cache="f16"):
    return {"n-gpu-layers": ngl, "cache-type-k": cache}


class TestVramPrecheck(unittest.TestCase):
    def test_uncalibrated_never_blocks(self):
        """未校准一律放行：错误的猜测式拦截比放行更糟（旧版就是这么坏的）"""
        v = _Vram(model_size_gb=40.0, gpu_vram_gb=24.0)
        for ctx in (8192, 65536, 131072):
            self.assertTrue(_fits_vram(_cfg(), v, ctx))

    def test_no_gpu_info_never_blocks(self):
        v = _Vram(model_size_gb=10.0, gpu_vram_gb=0.0)
        v.kv_bytes_per_token = 100000.0
        v.calibrated_ctx = 8192
        self.assertTrue(_fits_vram(_cfg(), v, 131072))

    def test_cpu_fallback_always_fits(self):
        v = _Vram(model_size_gb=40.0, gpu_vram_gb=24.0)
        v.kv_bytes_per_token = 500000.0
        v.calibrated_ctx = 8192
        self.assertTrue(_fits_vram(_cfg(ngl="0"), v, 131072))

    def test_estimate_scales_with_ctx(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0,
                  kv_bytes_per_token=131072.0, calibrated_ctx=8192)
        e8 = v.estimate_gb(8192, "f16")
        e32 = v.estimate_gb(32768, "f16")
        self.assertAlmostEqual(e8, 16.0 + 131072.0 * 8192 / GB, places=4)
        self.assertGreater(e32, e8)
        # ctx 翻 4 倍，KV 部分也翻 4 倍
        kv8 = (e8 - 16.0) * 4
        self.assertAlmostEqual(e32 - 16.0, kv8, places=4)

    def test_estimate_scales_with_cache_dtype(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0,
                  kv_bytes_per_token=131072.0, calibrated_ctx=8192)
        f16 = v.estimate_gb(32768, "f16")
        q8 = v.estimate_gb(32768, "q8_0")
        q4 = v.estimate_gb(32768, "q4_0")
        self.assertGreater(f16, q8)
        self.assertGreater(q8, q4)
        kv_f16 = f16 - 16.0
        self.assertAlmostEqual(q8 - 16.0, kv_f16 / 2, places=4)
        self.assertAlmostEqual(q4 - 16.0, kv_f16 / 4, places=4)

    def test_blocks_when_calibrated_over_budget(self):
        """校准过且确实放不下才拦；余量 6%"""
        # 18 GB 权重 + 393216 B/token * 8192 ≈ 3 GB KV = 21 GB，24 GB 卡放得下
        v = _Vram(model_size_gb=18.0, gpu_vram_gb=24.0,
                  kv_bytes_per_token=393216.0, calibrated_ctx=8192)
        self.assertTrue(_fits_vram(_cfg(), v, 8192))
        # 131072 上下文时 KV 约 48 GB，必超
        self.assertFalse(_fits_vram(_cfg(), v, 131072))

    def test_fits_vram_boundary_uses_94_percent(self):
        v = _Vram(model_size_gb=22.0, gpu_vram_gb=24.0)
        v.kv_bytes_per_token = 0.0
        # 手工把关卡推到刚好边界：估算 = 22 + 0.5 = 22.5
        v.kv_bytes_per_token = 0.5 * GB / 8192
        v.calibrated_ctx = 8192
        self.assertTrue(_fits_vram(_cfg(), v, 8192))   # 22.5 <= 22.56
        v.kv_bytes_per_token = 1.0 * GB / 8192
        self.assertFalse(_fits_vram(_cfg(), v, 8192))  # 23.0 > 22.56


class TestCalibrate(unittest.TestCase):
    def test_subtracts_idle_and_model(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0, idle_used_gb=2.0)
        # 实测 22 GB = 2(底噪) + 16(权重) + 4(KV)  → 4 GB / 8192 token
        self.assertTrue(_calibrate(v, 22.0, 8192, "f16"))
        self.assertTrue(v.calibrated)
        self.assertAlmostEqual(v.kv_bytes_per_token, 4 * GB / 8192, places=2)
        self.assertEqual(v.calibrated_ctx, 8192)

    def test_q8_cache_halves_bytes_per_token(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0, idle_used_gb=0.0)
        self.assertTrue(_calibrate(v, 20.0, 8192, "q8_0"))   # KV 4 GB，但 cache 只有一半
        self.assertAlmostEqual(v.kv_bytes_per_token, 8 * GB / 8192, places=2)
        # 换算回同一 ctx 的估算，应回到实测的 4 GB
        self.assertAlmostEqual(v.estimate_gb(8192, "q8_0") - 16.0, 4.0, places=3)

    def test_rejects_meaningless_reading(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0, idle_used_gb=1.0)
        self.assertFalse(_calibrate(v, 0.0, 8192, "f16"))     # 读不到显存
        self.assertFalse(_calibrate(v, 15.0, 8192, "f16"))    # 比权重还小
        self.assertFalse(_calibrate(v, 20.0, 0, "f16"))       # ctx 非法
        self.assertFalse(v.calibrated)


class TestScore(unittest.TestCase):
    def test_zero_decode_never_scores(self):
        """纯 CPU 组 decode 0.0 / prefill 255，旧版靠 prefill 抢到最高分（issue #5）"""
        self.assertEqual(_score({"decode": 0.0, "prefill": 255.0, "ttft_ms": 900}, "latency"), 0.0)
        self.assertEqual(_score({"decode": 0, "prefill": 5000.0}, "throughput"), 0.0)

    def test_normal_metrics_score_positive(self):
        self.assertGreater(_score({"decode": 30.0, "prefill": 900.0,
                                   "ttft_ms": 200}, "latency"), 0.0)


class TestI18nKeys(unittest.TestCase):
    def test_all_tuner_keys_bilingual(self):
        """tuner.py 里引用到的 L() 键必须在 i18n 里中英齐备"""
        src = TUNER_SRC.read_text(encoding="utf-8")
        keys = sorted(set(re.findall(r'L\(\s*"(tune\.[a-zA-Z0-9_.]+)"', src)))
        self.assertGreater(len(keys), 20, "没提取到 tuner 的 i18n 键，正则可能失效")
        set_lang("zh")
        zh = {k: L(k) for k in keys}
        set_lang("en")
        en = {k: L(k) for k in keys}
        set_lang("zh")
        for k in keys:
            self.assertNotEqual(zh[k], k, f"{k} 缺少中文翻译")
            self.assertNotEqual(en[k], k, f"{k} 缺少英文翻译")
            self.assertNotEqual(zh[k], en[k], f"{k} 中英译文相同，可能没翻译")

    def test_placeholders_match_between_locales(self):
        from app.services.i18n import MESSAGES
        for key, val in MESSAGES.items():
            if not key.startswith("tune."):
                continue
            zh_ph = set(re.findall(r"\{(\w+)\}", val.get("zh", "")))
            en_ph = set(re.findall(r"\{(\w+)\}", val.get("en", "")))
            self.assertEqual(zh_ph, en_ph, f"{key} 中英占位符不一致: {zh_ph} vs {en_ph}")


if __name__ == "__main__":
    unittest.main()
