"""目标机器配置 API

用户在此添加/管理本机或局域网内的目标机器，指定系统类型、引擎路径、
模型目录、端口。所有功能基于这些配置运行。
"""

import platform

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional


def _detect_local_os() -> str:
    """识别运行后端的本机操作系统：darwin->macos, windows->windows, 其余->linux"""
    s = platform.system().lower()
    if s == "darwin":
        return "macos"
    if s == "windows":
        return "windows"
    return "linux"

from ..models.target import (
    Target, load_targets, upsert_target, delete_target, get_target,
)
from ..services.executor import make_executor
from ..services.collectors import detect_hardware
from ..services import installer

router = APIRouter()


class TargetRequest(BaseModel):
    conn_type: str = "local"
    host: str = ""
    port: int = 22
    user: str = ""
    auth_type: str = "key"
    key_path: str = ""
    password: str = ""
    os: str = "linux"
    engine_type: str = "llama_cpp"
    engine_path: str = ""
    models_dir: str = ""
    service_port: int = 8080
    id: Optional[str] = None
    name: str = "本机"


@router.get("")
def list_targets():
    """列出所有已配置的目标机器"""
    return {"targets": [t.to_dict() for t in load_targets()]}


@router.post("")
def create_target(req: TargetRequest):
    """新增或更新目标机器"""
    data = req.model_dump()
    if not data.get("id"):
        data.pop("id", None)
    target = Target(**data)
    targets = upsert_target(target)
    return {"ok": True, "id": target.id, "targets": [t.to_dict() for t in targets]}


@router.delete("/{target_id}")
def remove_target(target_id: str):
    """删除目标机器"""
    targets = delete_target(target_id)
    return {"ok": True, "targets": [t.to_dict() for t in targets]}


@router.post("/test")
def test_connection(req: TargetRequest):
    """测试连接并返回硬件信息（不落盘）"""
    data = req.model_dump()
    data.pop("id", None)
    data.pop("name", None)
    target = Target(**data)
    executor = make_executor(target)
    try:
        # 先测基本连通性
        probe = "echo OK" if target.os != "windows" else 'echo OK'
        r = executor.run(probe, timeout=10)
        if not r.ok and "OK" not in r.stdout:
            return {"ok": False, "message": f"连接失败: {r.stderr or r.stdout}"}
        hw = detect_hardware(executor, target)
        return {"ok": True, "message": "连接成功", "hardware": hw}
    except Exception as e:
        return {"ok": False, "message": f"连接异常: {e}"}
    finally:
        executor.close()



@router.get("/{target_id}/engine")
def check_engine(target_id: str):
    """检测目标机是否已安装推理引擎"""
    target = get_target(target_id)
    if not target:
        return {"installed": False, "reason": "目标机器不存在"}
    executor = make_executor(target)
    try:
        return installer.detect_engine(executor, target)
    finally:
        executor.close()


class InstallRequest(BaseModel):
    target_id: str


@router.post("/install-engine")
def install_engine(req: InstallRequest):
    """启动一键安装推理引擎（后台任务，返回 job_id 供轮询）"""
    target = get_target(req.target_id)
    if not target:
        return {"ok": False, "message": "目标机器不存在"}
    job_id = installer.start_install(target)
    return {"ok": True, "job_id": job_id}


@router.get("/install-status/{job_id}")
def install_status(job_id: str):
    """查询安装任务状态与日志"""
    job = installer.get_job(job_id)
    if not job:
        return {"status": "not_found", "logs": []}
    return job



@router.get("/local-os")
def local_os():
    """返回运行后端的本机操作系统，供前端「本机」模式自动识别，无需用户手选"""
    return {"os": _detect_local_os()}



@router.get("/engines")
def list_engines():
    """返回所有可用推理引擎的元信息（名称/支持平台/模型格式/安装提示），供前端动态渲染"""
    from ..services.engine_registry import list_engines as _list
    return {"engines": _list()}
