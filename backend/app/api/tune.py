"""智能调优 API

启动两阶段调优压测任务（后台执行，返回 job_id），轮询查询进度与结果。
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict

from ..services import tuner

router = APIRouter()


class TuneRequest(BaseModel):
    target_id: str
    model: str
    ctx_size: int = 8192                 # 用户固定的上下文长度（约束，不被优化）
    goal: str = "latency"                # latency | throughput | prefill
    baseline_cfg: Optional[Dict] = None  # 用户当前参数，作为基线先测对比
    model_size_gb: float = 0.0           # 模型权重 GB，0 则后端自动探测


@router.post("/start")
def start(req: TuneRequest):
    """启动两阶段调优任务"""
    return tuner.start_tune(
        req.target_id, req.model, req.ctx_size, req.goal,
        req.baseline_cfg, req.model_size_gb,
    )


@router.get("/status/{job_id}")
def status(job_id: str):
    """查询调优进度与结果"""
    job = tuner.get_job(job_id)
    if not job:
        return {"status": "not_found", "logs": [], "results": []}
    return job


@router.get("/active")
def active(target_id: str):
    """返回该目标机正在运行的调优任务，供前端刷新后恢复轮询"""
    return {"jobs": tuner.list_active_jobs(target_id)}


@router.get("/options")
def options():
    """返回可选的优化目标与基线参数取值范围，供前端渲染"""
    return {
        "goals": [{"value": k, "label": v} for k, v in tuner.GOAL_LABELS.items()],
        "spec_options": tuner.SPEC_OPTIONS,
        "cache_options": tuner.CACHE_OPTIONS,
        "ngl_options": tuner.NGL_OPTIONS,
    }
