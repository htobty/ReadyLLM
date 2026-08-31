"""模型下载服务

在目标机上后台下载 GGUF 模型到其 models_dir，支持断点续传。
下载为耗时操作，采用后台线程执行 + 轮询已落盘字节计算进度。

跨平台：
  - Windows：curl.exe（Win10 1803+ 自带）断点续传
  - macOS / Linux：curl -C -
"""

import threading
import time
import uuid
from typing import Optional, List

from .executor import Executor
from .model_catalog import ModelEntry, get_by_id
from ..models.target import Target, get_target

# 任务表：job_id -> {...}
_JOBS: dict = {}
_LOCK = threading.Lock()


def _file_size(executor: Executor, target: Target, path: str) -> int:
    """查询目标机上文件当前字节数，不存在返回 0"""
    if target.os == "windows":
        cmd = (f'powershell -Command "if(Test-Path \'{path}\')'
               f'{(chr(123))}(Get-Item \'{path}\').Length{(chr(125))}else{{0}}"')
        result = executor.run(cmd, timeout=10)
    else:
        cmd = f'stat -c %s "{path}" 2>/dev/null || echo 0'
        result = executor.run(cmd, timeout=10)
    digits = "".join(c for c in result.stdout if c.isdigit())
    return int(digits) if digits else 0


def _remote_total_size(executor: Executor, url: str) -> int:
    """通过 HEAD 请求获取远端文件总大小（在目标机执行，走目标机网络）"""
    cmd = f'curl -sIL --max-time 20 "{url}" | grep -i content-length | tail -1'
    result = executor.run(cmd, timeout=25)
    digits = "".join(c for c in result.stdout if c.isdigit())
    return int(digits) if digits else 0


def _append_log(job_id: str, msg: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})


def get_job(job_id: str) -> Optional[dict]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> List[dict]:
    with _LOCK:
        return [
            {"job_id": j["job_id"], "model_id": j["model_id"],
             "status": j["status"], "target_id": j["target_id"],
             "downloaded": j["downloaded"], "total": j["total"]}
            for j in _JOBS.values()
        ]


def start_download(target_id: str, model_id: str, source: str = "hf-mirror") -> dict:
    """启动下载任务，返回 {ok, job_id} 或 {ok:False, message}
    source: huggingface | hf-mirror | modelscope（魔搭无对应仓库自动回退镜像站）"""
    target = get_target(target_id)
    if not target:
        return {"ok": False, "message": "目标机器不存在"}
    if not target.models_dir:
        return {"ok": False, "message": "目标机器未配置模型目录"}
    entry = get_by_id(model_id)
    if not entry:
        return {"ok": False, "message": "模型不存在于目录"}

    url, used_source = entry.resolve(source)
    from .collectors import path_join
    dest = path_join(target, target.models_dir, entry.filename)

    # 同一目标机同文件正在下载则复用
    with _LOCK:
        for j in _JOBS.values():
            if j["target_id"] == target_id and j["dest"] == dest and j["status"] == "downloading":
                return {"ok": True, "job_id": j["job_id"], "reused": True}

        job_id = uuid.uuid4().hex[:8]
        _JOBS[job_id] = {
            "job_id": job_id,
            "model_id": model_id,
            "target_id": target_id,
            "dest": dest,
            "url": url,
            "source": used_source,
            "status": "downloading",
            "downloaded": 0,
            "total": 0,
            "error": "",
            "logs": [],
        }

    def _worker():
        executor = None
        try:
            from .executor import make_executor
            executor = make_executor(target)
            _append_log(job_id, f"开始下载 {entry.name} {entry.quant} → {dest}")

            total = _remote_total_size(executor, url)
            if total == 0 and entry.size_gb:
                # HEAD 拿不到（网络/重定向限制）时，回退用目录预估大小
                total = int(entry.size_gb * 1024 * 1024 * 1024)
                _append_log(job_id, "无法获取精确大小，使用预估大小计算进度")
            with _LOCK:
                _JOBS[job_id]["total"] = total
            _append_log(job_id, f"文件总大小: {round(total/1024/1024,1) if total else '未知'} MB")

            # 下载到 .part 临时文件，完成后重命名，避免未完成文件被误判为已下载
            part_path = dest + ".part"
            if target.os == "windows":
                dl_cmd = (f'curl.exe -L -C - --retry 3 --retry-delay 2 '
                          f'-o "{part_path}" "{url}"')
            else:
                dl_cmd = (f'curl -L -C - --retry 3 --retry-delay 2 '
                          f'-o "{part_path}" "{url}"')

            # 后台执行下载（不阻塞等待），随后轮询进度
            _append_log(job_id, "发起下载请求（断点续传）...")
            result = executor.run(dl_cmd, timeout=7200)

            final_size = _file_size(executor, target, part_path)
            with _LOCK:
                job = _JOBS[job_id]
                job["downloaded"] = final_size

            if result.ok and (total == 0 or final_size >= total):
                # 下载完成：重命名 .part → 最终文件名
                if target.os == "windows":
                    mv_cmd = f'move /Y "{part_path}" "{dest}"'
                else:
                    mv_cmd = f'mv -f "{part_path}" "{dest}"'
                mv_result = executor.run(mv_cmd, timeout=10)
                if mv_result.ok:
                    with _LOCK:
                        _JOBS[job_id]["status"] = "success"
                    _append_log(job_id, "✓ 下载完成")
                else:
                    with _LOCK:
                        _JOBS[job_id]["status"] = "failed"
                        _JOBS[job_id]["error"] = "文件重命名失败: " + (mv_result.stderr or "")
                    _append_log(job_id, f"✗ 重命名失败: {mv_result.stderr}")
            else:
                with _LOCK:
                    job = _JOBS[job_id]
                    job["status"] = "failed"
                    job["error"] = result.stderr or "下载未完成"
                _append_log(job_id, f"✗ 下载失败: {result.stderr}")
        except Exception as e:
            with _LOCK:
                job = _JOBS[job_id]
                job["status"] = "failed"
                job["error"] = str(e)
            _append_log(job_id, f"✗ 异常: {e}")
        finally:
            if executor:
                executor.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True, "job_id": job_id}


def query_progress(job_id: str) -> Optional[dict]:
    """查询进度：对 downloading 中的任务，实时读取目标机 .part 文件字节数"""
    job = get_job(job_id)
    if not job:
        return None
    if job["status"] == "downloading":
        target = get_target(job["target_id"])
        if target:
            executor = make_executor_cached(target)
            try:
                # 下载中文件在 .part 路径
                part_path = job["dest"] + ".part"
                size = _file_size(executor, target, part_path)
                with _LOCK:
                    _JOBS[job_id]["downloaded"] = size
                job["downloaded"] = size
            finally:
                executor.close()
    return job


def make_executor_cached(target: Target):
    """每次新建执行器（SSH 连接开销可接受，避免跨线程共享 paramiko client）"""
    from .executor import make_executor
    return make_executor(target)
