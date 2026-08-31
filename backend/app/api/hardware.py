"""硬件检测 API（基于用户配置的 Target）"""

from fastapi import APIRouter, HTTPException

from ..models.target import get_target
from ..services.executor import make_executor
from ..services.collectors import detect_hardware

router = APIRouter()


def _resolve(target_id: str):
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在，请先在设置中配置")
    return target, make_executor(target)


@router.get("/detect")
def api_detect(target_id: str):
    """检测目标机器硬件信息"""
    target, executor = _resolve(target_id)
    try:
        return detect_hardware(executor, target)
    finally:
        executor.close()


@router.get("/recommend")
def api_recommend(target_id: str):
    """根据目标机器硬件推荐模型"""
    target, executor = _resolve(target_id)
    try:
        hw = detect_hardware(executor, target)
    finally:
        executor.close()

    gpu = hw.get("gpu") or {}
    gpu_mem_gb = gpu.get("total_memory_gb", 0)
    recommendations = []

    if gpu_mem_gb >= 24:
        recommendations += [
            {"model": "Qwen3-27B-Q4_K_M", "size_gb": 16.3, "expected_speed": "15-25 t/s", "fit": "显存充足，可跑大模型"},
            {"model": "Qwen3-8B-Q8_0", "size_gb": 8.5, "expected_speed": "40-60 t/s", "fit": "速度优先，高质量量化"},
        ]
    elif gpu_mem_gb >= 16:
        recommendations += [
            {"model": "Qwen3-14B-Q4_K_M", "size_gb": 9.0, "expected_speed": "25-40 t/s", "fit": "平衡选择"},
            {"model": "Qwen3-8B-Q8_0", "size_gb": 8.5, "expected_speed": "40-60 t/s", "fit": "速度优先"},
        ]
    elif gpu_mem_gb >= 8:
        recommendations += [
            {"model": "Qwen3-8B-Q4_K_M", "size_gb": 5.0, "expected_speed": "30-50 t/s", "fit": "显存有限，推荐 8B"},
        ]
    elif gpu_mem_gb > 0:
        recommendations += [
            {"model": "Qwen3-4B-Q4_K_M", "size_gb": 2.5, "expected_speed": "50-80 t/s", "fit": "显存较小，推荐 4B"},
        ]

    return {"hardware": hw, "recommendations": recommendations}
