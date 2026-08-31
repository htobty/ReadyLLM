"""llama.cpp 引擎适配器

启动/停止/状态检测均基于用户配置的 Target 执行，按目标 OS 适配。
Windows 复用已验证的 base64+bat+schtasks 方案（解决中文路径编码与 schtasks 长度限制）。
Linux 用 nohup 后台启动。
"""

import base64

from .engine_adapter import EngineAdapter, StartParams
from .executor import Executor
from .collectors import path_join
from ..models.target import Target

# llama-server 默认推荐参数（通用，不含任何特定机器/模型路径）
DEFAULT_ARGS = [
    "--ctx-size", "8192",
    "--flash-attn", "on",
    "--n-gpu-layers", "999",
    "--host", "0.0.0.0",
]


class LlamaCppAdapter(EngineAdapter):
    def __init__(self, executor: Executor, target: Target):
        self.executor = executor
        self.target = target

    def name(self) -> str:
        return "llama_cpp"

    def check_installed(self) -> bool:
        exe = self.target.engine_path
        if not exe:
            return False
        if self.target.os == "windows":
            result = self.executor.run(f'if exist "{exe}" (echo FOUND)')
            return "FOUND" in result.stdout
        else:
            result = self.executor.run(f'test -f "{exe}" && echo FOUND')
            return "FOUND" in result.stdout

    def start(self, params: StartParams) -> tuple[bool, str]:
        args = params.extra_args or list(DEFAULT_ARGS)
        # 注入端口
        if "--port" not in " ".join(args):
            args = args + ["--port", str(self.target.service_port)]
        args_str = " ".join(args)
        exe = self.target.engine_path
        model_path = params.model_path

        if self.target.os == "windows":
            return self._start_windows(exe, model_path, args_str)
        return self._start_linux(exe, model_path, args_str)

    def _start_windows(self, exe: str, model_path: str, args_str: str) -> tuple[bool, str]:
        bat_content = f'@echo off\r\n"{exe}" --model "{model_path}" {args_str}\r\n'
        b64 = base64.b64encode(bat_content.encode("gbk")).decode("ascii")
        bat_path = r"C:\temp\llama_start.bat"

        write_cmd = (
            f'powershell -Command "'
            f"New-Item -Path C:\\temp -ItemType Directory -Force | Out-Null; "
            f"[IO.File]::WriteAllBytes('{bat_path}', [Convert]::FromBase64String('{b64}'))"
            f'"'
        )
        result = self.executor.run(write_cmd, timeout=15)
        if not result.ok:
            return False, f"写入启动脚本失败: {result.stdout} {result.stderr}"

        run_cmd = (
            'schtasks /create /tn LlamaServer /tr "%s" /sc once /st 00:00 /f '
            '&& schtasks /run /tn LlamaServer' % bat_path
        )
        result = self.executor.run(run_cmd, timeout=15)
        if not result.ok:
            return False, f"启动失败: {result.stdout} {result.stderr}"
        return True, "启动命令已发送"

    def _start_linux(self, exe: str, model_path: str, args_str: str) -> tuple[bool, str]:
        cmd = (
            f'nohup "{exe}" --model "{model_path}" {args_str} '
            f'> /tmp/llama_server.log 2>&1 &'
        )
        result = self.executor.run(cmd, timeout=15)
        if not result.ok:
            return False, f"启动失败: {result.stdout} {result.stderr}"
        return True, "启动命令已发送"

    def stop(self) -> tuple[bool, str]:
        if self.target.os == "windows":
            result = self.executor.run("taskkill /f /im llama-server.exe", timeout=10)
        else:
            result = self.executor.run("pkill -f llama-server", timeout=10)
        if result.ok:
            return True, "服务已停止"
        return False, f"停止结果: {result.stdout} {result.stderr}"

    def is_running(self) -> bool:
        if self.target.os == "windows":
            result = self.executor.run('tasklist /fi "imagename eq llama-server.exe" /fo csv /nh')
            return "llama-server.exe" in result.stdout
        else:
            result = self.executor.run("pgrep -f llama-server")
            return bool(result.stdout)

    def get_metrics_url(self) -> str:
        # metrics 端口仅目标机本地可访问，采集时在目标机 curl 此地址
        return f"http://127.0.0.1:{self.target.service_port}/metrics"
