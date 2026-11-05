"""llama.cpp 引擎适配器

启动/停止/状态检测均基于用户配置的 Target 执行，按目标 OS 适配。
Windows 复用已验证的 base64+bat+schtasks 方案（解决中文路径编码与 schtasks 长度限制）。
Linux 用 nohup 后台启动。
"""

import base64
import time
from typing import Optional, Set

from .engine_adapter import EngineAdapter, StartParams
from .executor import Executor
from .collectors import path_join
from ..models.target import Target
from .i18n import L

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
        # 自己启动的那个 llama-server 的 PID；None 表示没认出（未启动或启动失败）
        self._pid: Optional[int] = None

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

        # 启动前先记下已存在的 llama-server，启动后取差集认出自己起的那个（issue #6 问题 9）。
        # 只按进程名停服会连带杀掉用户自己的服务，所以必须知道自己的 PID。
        self._pid = None
        before = self._list_pids()

        if self.target.os == "windows":
            ok, msg = self._start_windows(exe, model_path, args_str)
        else:
            ok, msg = self._start_linux(exe, model_path, args_str)
        if ok:
            self._pid = self._new_pid(before)
        return ok, msg

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
            return False, L("engine.write_script_fail", err=f"{result.stdout} {result.stderr}")

        run_cmd = (
            'schtasks /create /tn LlamaServer /tr "%s" /sc once /st 00:00 /f '
            '&& schtasks /run /tn LlamaServer' % bat_path
        )
        result = self.executor.run(run_cmd, timeout=15)
        if not result.ok:
            return False, L("engine.start_fail", err=f"{result.stdout} {result.stderr}")
        return True, L("engine.start_sent")

    def _start_linux(self, exe: str, model_path: str, args_str: str) -> tuple[bool, str]:
        cmd = (
            f'nohup "{exe}" --model "{model_path}" {args_str} '
            f'> /tmp/llama_server.log 2>&1 &'
        )
        result = self.executor.run(cmd, timeout=15)
        if not result.ok:
            return False, L("engine.start_fail", err=f"{result.stdout} {result.stderr}")
        return True, L("engine.start_sent")

    def _list_pids(self) -> Set[int]:
        """目标机上当前所有 llama-server 进程的 PID 集合"""
        if self.target.os == "windows":
            cmd = ('powershell -NoProfile -Command "'
                   "Get-CimInstance Win32_Process | "
                   "Where-Object {$_.Name -eq 'llama-server.exe'} | "
                   'Select-Object -ExpandProperty ProcessId"')
        else:
            cmd = "pgrep -f llama-server"
        r = self.executor.run(cmd, timeout=15)
        pids: Set[int] = set()
        for tok in (r.stdout or "").split():
            if tok.strip().isdigit():
                pids.add(int(tok.strip()))
        return pids

    def _new_pid(self, before: Set[int]) -> Optional[int]:
        """启动后新增的那个 PID；服务是异步起的，最多等约 6 秒"""
        for _ in range(6):
            new = self._list_pids() - before
            if new:
                return max(new)
            time.sleep(1)
        return None

    def stop(self) -> tuple[bool, str]:
        # 优先按自己启动的 PID 停（issue #6 问题 9）：按名字用 taskkill /im、pkill -f
        # 会连带杀掉用户自己的 llama-server；也可能杀不干净，留下占着端口和显存的
        # 残留进程，后续启动就会"起了但不 ready"，静默挂死（真机 job 9fa06f07）。
        # 已知局限：用户在 tuner 之外自己跑着服务时 _pid 为 None，只能回落到按名字杀，
        # 这种场景需要在调用方先判断「是否已有别人在跑的服务」。
        if self._pid:
            if self.target.os == "windows":
                cmd = f"taskkill /F /T /PID {self._pid}"
            else:
                cmd = f"kill -9 {self._pid}"
            result = self.executor.run(cmd, timeout=10)
            self._pid = None
            if result.ok:
                return True, L("engine.stop_ok")
            return False, L("engine.stop_result", err=f"{result.stdout} {result.stderr}")

        if self.target.os == "windows":
            result = self.executor.run("taskkill /f /im llama-server.exe", timeout=10)
        else:
            result = self.executor.run("pkill -f llama-server", timeout=10)
        if result.ok:
            return True, L("engine.stop_ok")
        return False, L("engine.stop_result", err=f"{result.stdout} {result.stderr}")

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
