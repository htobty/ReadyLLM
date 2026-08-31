"""调优结果持久化

智能调优（tuner / ai_tuner）的推荐参数此前只存在内存 _JOBS，后端重启即丢失。
本模块把每次调优的「最优参数」按 (目标机, 模型) 落盘，供部署页作为默认参数回填。

存储：~/.model-deploy-assistant/tune_history.json
结构：{ "<target_id>::<model>": {params, ctx_size, source, score, ts} }
同一台机器同一个模型只保留最近一次（覆盖写）。

params 为扁平字典 {参数名: 值}，参数名不带 -- 前缀，不含 ctx-size（单独存）。
"""

import json
import os
import time
import threading
from typing import Optional

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".model-deploy-assistant")
HISTORY_FILE = os.path.join(CONFIG_DIR, "tune_history.json")

_LOCK = threading.Lock()


def _key(target_id: str, model: str) -> str:
    return f"{target_id}::{model}"


def _load() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(data: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, HISTORY_FILE)


def save_latest(
    target_id: str,
    model: str,
    ctx_size: int,
    params: dict,
    source: str = "tuner",
    score: float = 0.0,
) -> None:
    """记录某目标机+模型的最近一次调优最优参数（覆盖写）。

    Args:
        params: 扁平参数字典 {参数名: 值}，不含 ctx-size
        source: 'tuner'（自动调优）或 'ai_tuner'（AI 调优）
        score: 该配置的测速得分（t/s），供前端展示
    """
    if not params:
        return
    with _LOCK:
        data = _load()
        data[_key(target_id, model)] = {
            "params": {k: str(v) for k, v in params.items()},
            "ctx_size": int(ctx_size),
            "source": source,
            "score": round(float(score), 2),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save(data)


def get_latest(target_id: str, model: str) -> Optional[dict]:
    """取某目标机+模型最近一次调优参数。

    返回时把 ctx_size 并入 params（键 'ctx-size'），便于前端直接渲染完整命令行。
    无记录返回 None。
    """
    with _LOCK:
        rec = _load().get(_key(target_id, model))
    if not rec:
        return None
    params = dict(rec.get("params", {}))
    if rec.get("ctx_size"):
        params["ctx-size"] = str(rec["ctx_size"])
    return {
        "params": params,
        "ctx_size": rec.get("ctx_size", 0),
        "source": rec.get("source", ""),
        "score": rec.get("score", 0),
        "ts": rec.get("ts", ""),
    }


def list_history() -> dict:
    """返回全部历史（调试/展示用）"""
    with _LOCK:
        return _load()
