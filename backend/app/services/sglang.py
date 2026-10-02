"""SGLang 引擎适配器

SGLang 是高吞吐推理框架，通过 `sglang serve <model>` 启动 OpenAI 兼容服务。
与 llama.cpp / vLLM 的关键差异（依据官方文档 docs.sglang.io）：
  - 模型格式：HuggingFace 权重（HF model id 或本地权重目录），不是 GGUF
  - 启动命令：模型是位置参数，形如 `sglang serve MODEL --host 0.0.0.0 --port 30000`
  - 指标：默认不暴露，必须带 --enable-metrics 启动才有 /metrics
    （Prometheus 文本，指标前缀 sglang:，且带 model_name 标签）
  - 平台：官方安装说明面向 Linux + NVIDIA GPU，Windows 需 WSL2

所有命令基于用户配置的 Target 执行，不硬编码任何环境。
"""

from .engine_adapter import EngineAdapter, StartParams
from .executor import Executor
from ..models.target import Target
from .i18n import L

# SGLang 默认推荐启动参数（通用，不含任何特定机器/模型）
# --enable-metrics 是监控采集的前提：缺少它 /metrics 不会暴露指标
DEFAULT_ARGS = [
    "--host", "0.0.0.0",
    "--enable-metrics",
]

# Windows 拦截提示：文案走 i18n（key: detect.sglang_windows），按界面语言在调用时取


class SGLangAdapter(EngineAdapter):
    def __init__(self, executor: Executor, target: Target):
        self.executor = executor
        self.target = target

    def name(self) -> str:
        return "sglang"

    def _sglang_cmd(self) -> str:
        """sglang 可执行命令：优先用户配置的 engine_path，否则默认 PATH 中的 sglang"""
        return self.target.engine_path or "sglang"

    # ==================== 检测 ====================

    def check_installed(self) -> bool:
        if self.target.os == "windows":
            # Windows 原生不支持 SGLang
            return False
        result = self.executor.run(f"{self._sglang_cmd()} --version 2>&1", timeout=25)
        out = (result.stdout or "").lower()
        if "not found" in out or "no module" in out or "command not found" in out:
            return False
        return bool(result.ok and ("sglang" in out or any(c.isdigit() for c in out)))

    # ==================== 启动 / 停止 ====================

    def start(self, params: StartParams) -> tuple[bool, str]:
        if self.target.os == "windows":
            return False, L("detect.sglang_windows")

        args = list(params.extra_args) if params.extra_args else list(DEFAULT_ARGS)
        # 监控数据依赖 --enable-metrics，用户自定义参数里没给就补上
        if "--enable-metrics" not in args:
            args = args + ["--enable-metrics"]
        # 注入端口（sglang serve 用 --port）
        if "--port" not in args:
            args = args + ["--port", str(self.target.service_port)]
        args_str = " ".join(args)

        # model_path 对 SGLang 而言是 HF 模型 id 或本地权重目录（位置参数）
        model = params.model_path
        cmd = f'{self._sglang_cmd()} serve "{model}" {args_str}'

        # 后台启动，日志落盘
        run_cmd = f"nohup {cmd} > /tmp/sglang_server.log 2>&1 &"
        result = self.executor.run(run_cmd, timeout=20)
        if not result.ok:
            return False, L("engine.start_fail", err=f"{result.stdout} {result.stderr}")
        return True, L("engine.start_sent_weights")

    def stop(self) -> tuple[bool, str]:
        if self.target.os == "windows":
            return False, L("detect.sglang_windows")
        # sglang serve 会派生 scheduler / detokenizer 子进程，按名匹配一并结束
        result = self.executor.run("pkill -f 'sglang'", timeout=10)
        if result.ok:
            return True, L("engine.stop_ok")
        return False, L("engine.stop_result", err=f"{result.stdout} {result.stderr}")

    def is_running(self) -> bool:
        if self.target.os == "windows":
            return False
        result = self.executor.run("pgrep -f 'sglang'")
        return bool(result.stdout.strip())

    # ==================== 监控 ====================

    def get_metrics_url(self) -> str:
        # SGLang 在 --port 暴露 /metrics（需以 --enable-metrics 启动）
        return f"http://127.0.0.1:{self.target.service_port}/metrics"