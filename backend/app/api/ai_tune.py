"""AI Agent 调优 API

配置管理 + 启动 Agent 调优任务 + 轮询进度。
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict

from ..services import ai_tuner, tune_history

router = APIRouter()


class AIConfigRequest(BaseModel):
    api_url: str
    api_key: str = ""
    model_name: str = ""


class AITuneRequest(BaseModel):
    target_id: str
    model: str
    ctx_size: int = 8192
    goal: str = "latency"       # latency | throughput | prefill
    user_desc: str = ""         # 用户场景描述（可选）


@router.get("/config")
def get_config():
    """获取 AI 配置（api_key 脱敏）"""
    cfg = ai_tuner.get_config()
    masked = dict(cfg)
    if masked.get("api_key"):
        k = masked["api_key"]
        masked["api_key"] = k[:4] + "***" + k[-4:] if len(k) > 8 else "***"
    return masked


@router.put("/config")
def put_config(req: AIConfigRequest):
    """保存 AI 配置。

    防御「脱敏回显 + 全量覆盖」陷阱：get_config 返回的 api_key 是脱敏串
    （如 sk-w***LXcQ），前端会把它回填进表单。若用户只改了 url/model、
    没动 key 就提交，表单里的脱敏串会覆盖掉文件中的真实 key。
    因此当提交的 api_key 为空、含脱敏标记 *** 或与当前脱敏值一致时，
    保留文件里已有的真实 key，绝不用密文覆盖。
    """
    existing = ai_tuner.get_config()
    real_key = existing.get("api_key", "")
    masked = (real_key[:4] + "***" + real_key[-4:]) if len(real_key) > 8 else "***"
    new_key = req.api_key
    if not new_key or new_key == masked or "***" in new_key:
        new_key = real_key  # 用户没真正改 key，保留原值
    ai_tuner.save_config({
        "api_url": req.api_url,
        "api_key": new_key,
        "model_name": req.model_name,
    })
    return {"ok": True}


@router.post("/test-connection")
def test_connection(req: AIConfigRequest):
    """测试 LLM API 连通性"""
    return ai_tuner.test_connection({
        "api_url": req.api_url,
        "api_key": req.api_key,
        "model_name": req.model_name,
    })


@router.post("/start")
def start(req: AITuneRequest):
    """启动 AI Agent 调优任务"""
    return ai_tuner.start_ai_tune(
        req.target_id, req.model, req.ctx_size, req.goal, req.user_desc,
    )


@router.get("/status/{job_id}")
def status(job_id: str):
    """查询 AI 调优进度与结果"""
    job = ai_tuner.get_job(job_id)
    if not job:
        return {"status": "not_found", "logs": [], "rounds": []}
    return job


@router.get("/active")
def active(target_id: str):
    """返回该目标机正在运行的 AI 调优任务，供前端刷新后恢复轮询"""
    return {"jobs": ai_tuner.list_active_jobs(target_id)}



class SaveTuneRequest(BaseModel):
    target_id: str
    model: str
    ctx_size: int                        # 固定上下文长度，随参数一并保存
    params: Dict[str, str]               # 最优参数（扁平字典，不含 ctx-size）
    score: float = 0.0


@router.post("/save")
def save(req: SaveTuneRequest):
    """把 AI 调优的最终推荐参数（含固定 ctx_size）保存到该模型，
    作为部署页 default-args 的回填来源。用户在结果界面点「保存并应用」时调用。"""
    if not req.params:
        return {"ok": False, "message": "无参数可保存"}
    tune_history.save_latest(
        req.target_id, req.model, req.ctx_size, req.params,
        source="ai_tuner", score=req.score,
    )
    return {"ok": True, "message": f"已保存到 {req.model} 的部署参数（含 ctx={req.ctx_size}）"}
