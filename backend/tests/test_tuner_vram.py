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
    def test_fixed_overhead_cancelled_by_two_points(self):
        """本次 bug 回归：MTP draft / compute buffer 等固定开销不得混入 KV 斜率

        真机形态（27B Q4_K_M / 4090 / q4_0）：MTP 额外固定占 2.41 GB。
        旧单点算法把它一并除以 ctx，报出 1431306 B/token（虚高 5.46 倍）；
        两点差分把固定开销相减消掉，应回到 262144 B/token（f16 理论基准）。
        """
        kv_q4 = 65536                       # q4_0 每 token 字节（f16 的 1/4）
        fixed = 2.41                        # MTP 固定开销，不随 ctx 变化
        base = 0.42 + 16.64 + fixed         # idle + 权重 + 固定开销
        u1 = base + kv_q4 * 8192 / GB
        u2 = base + kv_q4 * 32768 / GB
        v = _Vram(model_size_gb=16.64, gpu_vram_gb=24.0, idle_used_gb=0.42)
        self.assertTrue(_calibrate(v, [(8192, u1), (32768, u2)], "q4_0"))
        self.assertTrue(v.calibrated)
        self.assertAlmostEqual(v.kv_bytes_per_token, 262144, delta=1)
        self.assertEqual(v.calibrated_ctx, 32768)
        # 关键后果：f16 KV@8192 应估成 18.64 GB，24 GB 卡上必须放行而不是被误拦
        self.assertAlmostEqual(v.estimate_gb(8192, "f16"), 18.64, delta=0.02)
        self.assertTrue(_fits_vram(_cfg(cache="f16"), v, 8192))

    def test_q4_calibration_scales_to_f16_basis(self):
        """q4_0 实测斜率换算成 f16 基准要乘 4"""
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0)
        u1 = 17.0
        u2 = u1 + (65536 * 16384) / GB      # ctx 多 16384 对应的 q4_0 KV
        self.assertTrue(_calibrate(v, [(8192, u1), (24576, u2)], "q4_0"))
        self.assertAlmostEqual(v.kv_bytes_per_token, 262144, delta=1)

    def test_rejects_meaningless_reading(self):
        v = _Vram(model_size_gb=16.0, gpu_vram_gb=24.0, idle_used_gb=1.0)
        self.assertFalse(_calibrate(v, [], "f16"))                              # 没有点
        self.assertFalse(_calibrate(v, [(8192, 20.0)], "f16"))                  # 只有一点
        self.assertFalse(_calibrate(v, [(8192, 0.0), (32768, 21.0)], "f16"))    # 读不到显存
        self.assertFalse(_calibrate(v, [(8192, 20.0), (32768, 20.05)], "f16"))  # 差值低于阈值
        self.assertFalse(_calibrate(v, [(8192, 20.0), (8192, 22.0)], "f16"))    # ctx 相同
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



class TestBenchFastFail(unittest.TestCase):
    """服务崩掉时必须快速失败，不能跑满 5 遍 bench × 2 个请求 × --max-time 120。

    真机 job 9fa06f07 实测：llama-server 进程还在、8989 已不监听，
    tuner 在 running 状态静默卡了 13 分钟以上，前端完全看不出服务已经死了。
    """

    DEAD = {"decode": 0.0, "prefill": 0.0, "ttft_ms": 0.0, "gpu_util": 0.0,
            "gpu_mem_pct": 0.0, "cpu_pct": 0.0, "mem_used_gb": 0.0,
            "mem_total_gb": 0.0, "mem_pct": 0.0}

    def test_bench_median_bails_out_when_service_dead(self):
        from unittest.mock import patch
        from app.services import tuner as T

        calls = []
        with patch.object(T, "_bench_once",
                          side_effect=lambda *a, **k: (calls.append(1), dict(self.DEAD))[1]):
            out = T._bench_median(None, None, 8192)

        self.assertEqual(out["decode"], 0.0)
        self.assertLessEqual(
            len(calls), T._BENCH_DEAD_STREAK,
            "服务已崩时不应继续跑剩余 bench 轮次",
        )
        self.assertLess(len(calls), T._BENCH_WARMUP + T._BENCH_REPEATS)

    def test_bench_median_still_runs_full_rounds_when_healthy(self):
        from unittest.mock import patch
        from app.services import tuner as T

        ok = dict(self.DEAD, decode=34.0, prefill=2583.0, ttft_ms=45.0)
        calls = []
        with patch.object(T, "_bench_once",
                          side_effect=lambda *a, **k: (calls.append(1), dict(ok))[1]):
            out = T._bench_median(None, None, 8192)

        self.assertEqual(len(calls), T._BENCH_WARMUP + T._BENCH_REPEATS)
        self.assertAlmostEqual(out["decode"], 34.0, places=2)

    def test_health_ok_reflects_port_state(self):
        from unittest.mock import MagicMock
        from app.services import tuner as T

        ex = MagicMock()
        tgt = MagicMock()
        tgt.service_port = 8989

        ex.run.return_value.stdout = "000"
        self.assertFalse(T._health_ok(ex, tgt), "端口无人监听时 health 必须为 False")

        ex.run.return_value.stdout = "200"
        self.assertTrue(T._health_ok(ex, tgt))



class TestParseLiveCfg(unittest.TestCase):
    """issue #6 问题 6：基线要能来自用户真实在跑的 llama-server 命令行。

    解析必须只取「影响速度」的开关；模型路径、ctx、端口、host、metrics 由 tuner
    自己接管，--fit / --flash-attn 属于显存与上下文策略（问题 8），基线里不覆盖。
    """

    def test_extracts_speed_keys_and_drops_plumbing(self):
        from app.services.tuner import _parse_llama_cmdline

        cmd = (
            'H:\\llama\\llama-server.exe --model F:\\models\\Qwen3.8-27B-Q4_K_M.gguf '
            '--ctx-size 8192 --n-gpu-layers all --cache-type-k q8_0 --cache-type-v q8_0 '
            '--batch-size 2048 --ubatch-size 512 --spec-type draft-mtp,ngram-mod '
            '--flash-attn on --fit on --metrics --host 0.0.0.0 --port 8989'
        )
        cfg = _parse_llama_cmdline(cmd)

        # 取到的
        self.assertEqual(cfg["n-gpu-layers"], "all")
        self.assertEqual(cfg["cache-type-k"], "q8_0")
        self.assertEqual(cfg["cache-type-v"], "q8_0")
        self.assertEqual(cfg["batch-size"], "2048")
        self.assertEqual(cfg["ubatch-size"], "512")
        # 复合 spec-type 必须原样保留成单值，不能按逗号拆开（问题 7 的前提）
        self.assertEqual(cfg["spec-type"], "draft-mtp,ngram-mod")

        # 丢掉的：这些由 tuner 接管，或属于显存策略
        for k in ("model", "ctx-size", "port", "host", "metrics", "fit", "flash-attn"):
            self.assertNotIn(k, cfg, f"{k} 不该进入基线配置")

    def test_short_aliases_map_to_long_keys(self):
        from app.services.tuner import _parse_llama_cmdline

        cfg = _parse_llama_cmdline("llama-server -ngl 99 -ctk q4_0 -ctv q4_0 -b 4096")
        self.assertEqual(cfg["n-gpu-layers"], "99")
        self.assertEqual(cfg["cache-type-k"], "q4_0")
        self.assertEqual(cfg["cache-type-v"], "q4_0")
        self.assertEqual(cfg["batch-size"], "4096")

    def test_empty_or_garbage_returns_empty_dict(self):
        """读不到命令行时必须返回空 dict，让调用方回落到内置基线而不是崩掉"""
        from app.services.tuner import _parse_llama_cmdline

        self.assertEqual(_parse_llama_cmdline(""), {})
        self.assertEqual(_parse_llama_cmdline("llama-server --model x.gguf"), {})



class TestStopByPid(unittest.TestCase):
    """issue #6 问题 9：停服要按自己启动的 PID，不能按进程名。

    按名 taskkill /im、pkill -f 会连带杀掉用户自己的 llama-server；也可能杀不干净，
    留下占着端口和显存的僵尸进程，后续步骤再启动就"起了但不 ready"，静默挂死
    （真机 job 9fa06f07 实测：8989 不监听、GPU 0%，job 却一直是 running）。
    """

    def _adapter(self, os_name="windows", stdout=""):
        from unittest.mock import MagicMock
        from app.services.llama_cpp import LlamaCppAdapter

        ex = MagicMock()
        ex.run.return_value.stdout = stdout
        ex.run.return_value.ok = True
        tgt = MagicMock()
        tgt.os = os_name
        return LlamaCppAdapter(ex, tgt), ex

    def test_stop_kills_own_pid_not_image_name(self):
        a, ex = self._adapter()
        a._pid = 5696
        ok, _ = a.stop()
        self.assertTrue(ok)
        cmd = ex.run.call_args[0][0]
        self.assertIn("5696", cmd, "应按记录的 PID 停服")
        self.assertNotIn("/im", cmd, "不能再按镜像名 taskkill")
        self.assertIsNone(a._pid, "停服后必须清掉 PID，避免下次误杀复用的进程号")

    def test_stop_falls_back_to_name_when_pid_unknown(self):
        """没认出 PID（例如服务不是我们启的）时保留旧行为，不能直接不干活"""
        a, ex = self._adapter()
        a._pid = None
        a.stop()
        cmd = ex.run.call_args[0][0]
        self.assertIn("/im", cmd)

    def test_list_pids_parses_windows_multiline(self):
        a, _ = self._adapter(stdout="1234\r\n5678\r\n")
        self.assertEqual(a._list_pids(), {1234, 5678})

    def test_list_pids_ignores_non_numeric(self):
        a, _ = self._adapter(stdout="No Process Found\n")
        self.assertEqual(a._list_pids(), set())



class TestArgsListSpecType(unittest.TestCase):
    """--spec-type 非法值回归。

    llama.cpp 的 --spec-type 合法值是 none / draft-simple / draft-eagle3 / draft-mtp /
    draft-dflash / draft-dspark / ngram-simple / ngram-map-k / ngram-map-k4v / ngram-mod /
    ngram-cache，**没有 off**。
    tuner 内部用 "off" 表示关闭投机解码，若原样拼进命令行，llama-server 会在参数解析阶段
    直接退出（unknown speculative type: off），_wait_ready 白等满超时，最后报出
    「启动超时(可能显存不足)」——一个完全错误的归因，而且会静默丢掉 half the search space。
    """

    class _T:  # target 只需要 service_port
        service_port = 8989

    def _args(self, cfg):
        from app.services.tuner import _args_list
        return _args_list(cfg, self._T(), 8192)

    def test_off_maps_to_none(self):
        cfg = {"spec-type": "off", "cache-type-k": "q4_0"}
        args = self._args(cfg)
        self.assertIn("--spec-type", args)
        idx = args.index("--spec-type")
        self.assertEqual(args[idx + 1], "none")
        # 注意：不能断言 "off" 整体不出现——--fit off 是固定项，合法且必须保留。
        # 只校验紧跟在 --spec-type 后面的那个值。
        self.assertNotEqual(args[idx + 1], "off")

    def test_draft_mtp_passed_through(self):
        cfg = {"spec-type": "draft-mtp"}
        args = self._args(cfg)
        idx = args.index("--spec-type")
        self.assertEqual(args[idx + 1], "draft-mtp")

    def test_compound_spec_type_passed_through(self):
        # issue #6 问题 7 的目标形态：复合 spec-type 必须能完整透传
        cfg = {"spec-type": "draft-mtp,ngram-mod"}
        args = self._args(cfg)
        idx = args.index("--spec-type")
        self.assertEqual(args[idx + 1], "draft-mtp,ngram-mod")

    def test_no_spec_type_key_omits_flag(self):
        args = self._args({"cache-type-k": "q4_0"})
        self.assertNotIn("--spec-type", args)



class TestWaitReadyDetectsProcessExit(unittest.TestCase):
    """启动失败要能被快速识别，而不是干等满超时后误报成显存不足。

    真机事故：--spec-type off 是非法值，llama-server 在参数解析阶段就退出，
    端口永远不通，_wait_ready 白等 120 秒，最终报「启动超时(可能显存不足)」。
    """

    class _T:
        service_port = 8989

    class _Resp:
        def __init__(self, out):
            self.stdout = out

    class _Exec:
        """永远返回 000（服务不监听）"""
        def run(self, cmd, timeout=None):
            return TestWaitReadyDetectsProcessExit._Resp("000")

    class _DeadEngine:
        _pid = 4242

        def _list_pids(self):
            return set()          # 进程已经没了

    class _AliveEngine:
        _pid = 4242

        def _list_pids(self):
            return {4242}         # 进程还在，只是还没就绪

    def _run(self, engine, timeout):
        import time as _t
        from app.services import tuner
        orig = tuner.time.sleep
        tuner.time.sleep = lambda s: None      # 去掉真实等待，避免单测跑满超时
        try:
            t0 = _t.time()
            ok = tuner._wait_ready(self._Exec(), self._T(), timeout=timeout, engine=engine)
            return ok, _t.time() - t0
        finally:
            tuner.time.sleep = orig

    def test_dead_process_fails_fast(self):
        ok, elapsed = self._run(self._DeadEngine(), timeout=120)
        self.assertFalse(ok)
        self.assertLess(elapsed, 30, "进程已退出时应立刻失败，不该等满 120 秒")

    def test_alive_process_still_waits_until_timeout(self):
        ok, _ = self._run(self._AliveEngine(), timeout=8)
        self.assertFalse(ok)

    def test_no_engine_keeps_backward_compatible_behavior(self):
        ok, _ = self._run(None, timeout=6)
        self.assertFalse(ok)
