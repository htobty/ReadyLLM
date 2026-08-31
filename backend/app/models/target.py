"""目标机器配置模型

面向所有用户的通用工具：目标机器、引擎路径、模型目录、端口
全部由用户配置，严禁硬编码任何特定环境。
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

# 配置持久化目录（用户主目录，非项目目录）
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".model-deploy-assistant")
CONFIG_FILE = os.path.join(CONFIG_DIR, "targets.json")


@dataclass
class Target:
    """目标机器配置"""
    # 连接方式：local（本机）或 ssh（远程）
    conn_type: str = "local"

    # SSH 连接参数（conn_type=ssh 时使用）
    host: str = ""
    port: int = 22
    user: str = ""
    auth_type: str = "key"          # key（密钥）或 password（密码）
    key_path: str = ""              # 私钥路径，默认 ~/.ssh/id_rsa
    password: str = ""              # 密码认证时使用

    # 目标系统类型：windows / linux
    os: str = "linux"

    # 推理引擎类型：llama_cpp / vllm（默认 llama_cpp，向后兼容旧配置）
    engine_type: str = "llama_cpp"

    # 推理引擎可执行文件路径或命令
    #   llama_cpp: llama-server.exe / /usr/local/bin/llama-server
    #   vllm:      vllm 命令（pip 安装后通常即在 PATH，可留空用默认）
    engine_path: str = ""

    # 模型目录（存放 .gguf 的目录）
    models_dir: str = ""

    # 推理服务监听端口
    service_port: int = 8080

    # 元信息
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "本机"

    def to_dict(self) -> dict:
        d = asdict(self)
        # 不持久化明文密码
        d["password"] = ""
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        # 过滤掉不属于 dataclass 的字段
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ==================== 持久化 ====================

def _ensure_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


def load_targets() -> list[Target]:
    """加载所有已保存的目标机器"""
    if not os.path.exists(CONFIG_FILE):
        return []
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Target.from_dict(t) for t in data.get("targets", [])]
    except Exception:
        return []


def save_targets(targets: list[Target]) -> None:
    """保存目标机器列表"""
    _ensure_dir()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"targets": [t.to_dict() for t in targets]},
            f, ensure_ascii=False, indent=2,
        )


def get_target(target_id: str) -> Optional[Target]:
    """按 id 获取单个目标机器"""
    for t in load_targets():
        if t.id == target_id:
            return t
    return None


def upsert_target(target: Target) -> list[Target]:
    """新增或更新一个目标机器"""
    targets = load_targets()
    for i, t in enumerate(targets):
        if t.id == target.id:
            targets[i] = target
            break
    else:
        targets.append(target)
    save_targets(targets)
    return targets


def delete_target(target_id: str) -> list[Target]:
    """删除一个目标机器"""
    targets = [t for t in load_targets() if t.id != target_id]
    save_targets(targets)
    return targets
