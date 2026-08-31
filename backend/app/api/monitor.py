"""监控 API + WebSocket 推送（基于用户配置的 Target）"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from ..config import REFRESH_INTERVAL
from ..models.target import get_target
from ..services.executor import make_executor
from ..services.collectors import collect_all

router = APIRouter()


@router.get("/snapshot")
def get_snapshot(target_id: str):
    """获取一次监控快照"""
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在，请先在设置中配置")
    executor = make_executor(target)
    try:
        return collect_all(executor, target)
    finally:
        executor.close()


@router.websocket("/ws")
async def monitor_websocket(ws: WebSocket, target_id: str = ""):
    """WebSocket 实时推送监控数据"""
    await ws.accept()

    target = get_target(target_id)
    if not target:
        await ws.send_text(json.dumps({"error": "目标机器不存在"}))
        await ws.close()
        return

    executor = make_executor(target)
    try:
        while True:
            data = await asyncio.to_thread(collect_all, executor, target)
            await ws.send_text(json.dumps(data, ensure_ascii=False))
            await asyncio.sleep(REFRESH_INTERVAL / 1000)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        executor.close()
