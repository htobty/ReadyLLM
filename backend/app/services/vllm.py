"""vLLM 引擎适配器

vLLM 是高吞吐推理引擎，通过 `vllm serve <model>` 启动 OpenAI 兼容服务。
与 llama.cpp 的关键差异：
  - 模型格式：HuggingFace safetensors（用 HF 模型 id 或本地权重目录），不是 GGUF
  - 安装方式：pip install vllm（依赖 Python + CUDA），不是下载二进制
  - 平台限制：不支持 Windows 原生运行，只能在 Linux / macOS 或 WSL2 中部署

因此本适配器在 Windows 目标机上不尝试启动，而是返回明确的 WSL2 引导提示。
所有命令基于用户配置的 Target 执行，不硬编码任何环境。
"""

from .engine_adapter import EngineAdapter, StartParams
from .executor import Executor
from ..models.target import Target

# vLLM 默认推荐启动参数（通用，不含任何特定机器/模型）
DEFAULT_ARGS = [
    "--max-model-len", "8192",
    "--host", "0.0.0.0",
]

WSL2_HINT = (
    "vLLM 不支持 Windows 原生运行。请在 WSL2 (Ubuntu) 中安装并运行 vLLM，"
    "或将目标机引擎类型改为 llama.cpp。"
)


class VLLMAdapter(EngineAdapter):
    def __init__(self, executor: Executor, target: Target):
        self.executor = executor
        self.target = target

    def name(self) -> str:
        return "vllm"

    def _vllm_cmd(self) -> str:
        """vllm 可执行命令：优先用户配置的 engine_path，否则默认 PATH 中的 vllm"""
        return self.target.engine_path or "vllm"

    # ==================== 检测 ====================

    def check_installed(self) -> bool:
        if self.target.os == "windows":
            # Windows 原生不支持 vLLM
            return False
        result = self.executor.run(f"{self._vllm_cmd()} --version 2>&1", timeout=20)
        out = (result.stdout or "").lower()
        # vllm --version 会打印版本号；未安装时报 command not found / no module
        if result.ok and ("vllm" in out or any(c.isdigit() for c in out)):
            if "not found" not in out and "no module" not in out and "error" not in out.split("version")[0]:
                return True
        return False

    # ==================== 启动 / 停止 ====================

    def start(self, params: StartParams) -> tuple[bool, str]:
        if self.target.os == "windows":
            return False, WSL2_HINT

        args = list(params.extra_args) if params.extra_args else list(DEFAULT_ARGS)
        # 注入端口（vllm serve 用 --port）
        if "--port" not in " ".join(args):
            args = args + ["--port", str(self.target.service_port)]
        args_str = " ".join(args)
        # model_path 对 vLLM 而言是 HF 模型 id 或本地 safetensors 目录
        model = params.model_path
        cmd = f'{self._vllm_cmd()} serve "{model}" {args_str}'

        # 后台启动，日志落盘
        run_cmd = f"nohup {cmd} > /tmp/vllm_server.log 2>&1 &"
        result = self.executor.run(run_cmd, timeout=20)
        if not result.ok:
            return False, f"启动失败: {result.stdout} {result.stderr}"
        return True, "vLLM 启动命令已发送（首次加载模型需下载权重，请耐心等待）"

    def stop(self) -> tuple[bool, str]:
        if self.target.os == "windows":
            return False, WSL2_HINT
        result = self.executor.run("pkill -f 'vllm serve'", timeout=10)
        if result.ok:
            return True, "vLLM 服务已停止"
        return False, f"停止结果: {result.stdout} {result.stderr}"

    def is_running(self) -> bool:
        if self.target.os == "windows":
            return False
        result = self.executor.run("pgrep -f 'vllm serve'")
        return bool(result.stdout.strip())

    # ==================== 监控 ====================

    def get_metrics_url(self) -> str:
        # vLLM 默认在 --port 暴露 /metrics（Prometheus 格式）
        return f"http://127.0.0.1:{self.target.service_port}/metrics"
