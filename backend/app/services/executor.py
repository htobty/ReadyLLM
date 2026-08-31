"""执行器抽象层

统一本机执行与远程 SSH 执行接口，所有功能基于用户配置的 Target 运行，
不依赖任何硬编码环境。

除命令行执行 run() 外，另提供 write_file()：通过 SFTP（远程）或本地文件
写入大体积内容，绕开 Windows cmd.exe 命令行 8191 字符上限——长 prompt 测速
必须走此通道，否则 base64 内嵌的命令会被截断。
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from ..models.target import Target


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _decode(data: bytes) -> str:
    """尝试 UTF-8 解码，失败回退 GBK（Windows 中文系统常见）"""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="replace")


class Executor:
    """执行器基类"""

    def run(self, cmd: str, timeout: int = 15) -> ExecResult:
        raise NotImplementedError

    def write_file(self, content: str, path: str) -> bool:
        """把文本内容以 UTF-8 写入目标机的 path（不受命令行长度限制）"""
        raise NotImplementedError

    def read_file_bytes(self, path: str) -> Optional[bytes]:
        """读取目标机上 path 的二进制内容（用于把成片等产物拉回控制端）。失败返回 None。"""
        raise NotImplementedError

    def write_file_bytes(self, data: bytes, path: str) -> bool:
        """把二进制内容写入目标机的 path（用于上传首帧图等产物）。成功返回 True。"""
        raise NotImplementedError

    def close(self):
        pass


class LocalExecutor(Executor):
    """本机执行器"""

    def run(self, cmd: str, timeout: int = 15) -> ExecResult:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=timeout
            )
            return ExecResult(
                stdout=_decode(result.stdout).strip(),
                stderr=_decode(result.stderr).strip(),
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="命令执行超时", returncode=-1)
        except Exception as e:
            return ExecResult(stdout="", stderr=str(e), returncode=-1)

    def write_file(self, content: str, path: str) -> bool:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "wb") as f:
                f.write(content.encode("utf-8"))
            return True
        except Exception:
            return False

    def read_file_bytes(self, path: str) -> Optional[bytes]:
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def write_file_bytes(self, data: bytes, path: str) -> bool:
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            return False


class SSHExecutor(Executor):
    """远程 SSH 执行器（paramiko）"""

    def __init__(self, target: Target):
        self.target = target
        self._client = None

    def _get_client(self):
        import paramiko

        if self._client is not None:
            return self._client

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        t = self.target
        connect_kwargs = {
            "hostname": t.host,
            "port": t.port,
            "username": t.user,
            "timeout": 10,
        }

        if t.auth_type == "password" and t.password:
            connect_kwargs["password"] = t.password
        else:
            # 密钥认证，默认 ~/.ssh/id_rsa
            key = t.key_path or os.path.expanduser("~/.ssh/id_rsa")
            if os.path.exists(key):
                connect_kwargs["key_filename"] = key

        client.connect(**connect_kwargs)
        self._client = client
        return client

    def run(self, cmd: str, timeout: int = 15) -> ExecResult:
        try:
            client = self._get_client()
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            out = _decode(stdout.read()).strip()
            err = _decode(stderr.read()).strip()
            rc = stdout.channel.recv_exit_status()
            return ExecResult(stdout=out, stderr=err, returncode=rc)
        except Exception as e:
            # 连接异常时重置客户端，下次重连
            self._client = None
            return ExecResult(stdout="", stderr=str(e), returncode=-1)

    def write_file(self, content: str, path: str) -> bool:
        """通过 SFTP 写文件，绕开命令行长度限制。

        Windows OpenSSH 的 SFTP 接受正斜杠路径（如 C:/temp/bench.json），
        目录需调用方先行创建（SFTP 跨平台逐级建目录不可靠）。
        """
        try:
            client = self._get_client()
            sftp = client.open_sftp()
            try:
                with sftp.file(path, "wb") as f:
                    f.write(content.encode("utf-8"))
            finally:
                sftp.close()
            return True
        except Exception:
            self._client = None
            return False

    def read_file_bytes(self, path: str) -> Optional[bytes]:
        """通过 SFTP 读取目标机文件的二进制内容（如把成片 mp4 拉回控制端）。
        Windows OpenSSH 的 SFTP 接受正斜杠路径，调用方需先把反斜杠转过来。"""
        try:
            client = self._get_client()
            sftp = client.open_sftp()
            try:
                with sftp.open(path.replace("\\", "/"), "rb") as f:
                    return f.read()
            finally:
                sftp.close()
        except Exception:
            self._client = None
            return None

    def write_file_bytes(self, data: bytes, path: str) -> bool:
        """通过 SFTP 上传二进制文件到目标机（如把首帧图推到 ComfyUI/input）。
        Windows OpenSSH 的 SFTP 接受正斜杠路径，调用方需先把反斜杠转过来。"""
        try:
            client = self._get_client()
            sftp = client.open_sftp()
            try:
                with sftp.open(path.replace("\\", "/"), "wb") as f:
                    f.write(data)
            finally:
                sftp.close()
            return True
        except Exception:
            self._client = None
            return False

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


def make_executor(target: Target) -> Executor:
    """根据 Target 配置创建对应执行器"""
    if target.conn_type == "ssh":
        return SSHExecutor(target)
    return LocalExecutor()
