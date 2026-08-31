"""模型商店 API

提供内置模型目录浏览、按目标机显存筛选、启动下载、进度查询、
已下载模型列表。下载在目标机执行，支持断点续传。
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..services.model_catalog import list_all, filter_by_vram, get_by_id, fetch_dynamic_catalog
from ..services import downloader
from ..models.target import get_target
from ..services.executor import make_executor

router = APIRouter()


@router.get("/jobs")
def active_jobs(target_id: Optional[str] = None):
    """列出所有下载任务（含进行中的），前端刷新后据此恢复进度"""
    jobs = downloader.list_jobs()
    if target_id:
        jobs = [j for j in jobs if j["target_id"] == target_id]
    return {"jobs": jobs}


@router.get("/models")
def models(target_id: Optional[str] = None, source: str = "hf-mirror",
           category: Optional[str] = None):
    """模型列表；category=text/video 按类别筛选；给定 target_id 时按该机显存
    标注是否可跑，source 选择下载源"""
    items = list_all(source, category)
    if target_id:
        target = get_target(target_id)
        vram = 0.0
        if target:
            executor = make_executor(target)
            try:
                from ..services.collectors import _detect_gpu_static, _detect_memory_static
                gpu = _detect_gpu_static(executor, target)
                if gpu:
                    vram = gpu.get("total_memory_gb", 0)
                    # Apple Silicon 统一内存：可用显存≈物理内存的 75%（Metal 默认上限）
                    if gpu.get("unified") or vram == 0:
                        mem = _detect_memory_static(executor, target)
                        vram = mem.get("total_gb", 0) * 0.75
            finally:
                executor.close()
        for it in items:
            it["fits"] = it["min_vram_gb"] <= vram if vram > 0 else None
    return {"models": items}


@router.get("/refresh")
def refresh_catalog():
    """手动刷新：从 HuggingFace 动态获取最新热门 GGUF 模型"""
    result = fetch_dynamic_catalog(force=True)
    return result


@router.get("/dynamic")
def dynamic_models():
    """获取动态模型列表（有缓存则用缓存）"""
    result = fetch_dynamic_catalog(force=False)
    return result


class DownloadRequest(BaseModel):
    target_id: str
    model_id: str
    source: str = "hf-mirror"  # huggingface | hf-mirror | modelscope


@router.post("/download")
def download(req: DownloadRequest):
    """启动下载任务（source 选择下载源，魔搭无对应仓库自动回退镜像站）"""
    return downloader.start_download(req.target_id, req.model_id, req.source)


@router.get("/download/{job_id}")
def download_status(job_id: str):
    """查询下载进度（实时读取目标机已落盘字节）"""
    job = downloader.query_progress(job_id)
    if not job:
        return {"status": "not_found", "logs": []}
    total = job.get("total", 0)
    done = job.get("downloaded", 0)
    job["percent"] = round(done / total * 100, 1) if total > 0 else 0
    return job


@router.get("/downloaded")
def downloaded(target_id: str):
    """列出目标机模型目录下的 .gguf 文件（含实际大小），前端据此判断完整性"""
    target = get_target(target_id)
    if not target or not target.models_dir:
        return {"models": [], "error": "未配置模型目录" if not (target and target.models_dir) else "目标机器不存在"}
    executor = make_executor(target)
    try:
        if target.os == "windows":
            # 输出 "文件名|字节数" 格式
            cmd = (f'powershell -Command "Get-ChildItem \'{target.models_dir}\\*.gguf\' '
                   f'| ForEach-Object {{ $_.Name + \'|\' + $_.Length }}"')
            result = executor.run(cmd, timeout=15)
        else:
            result = executor.run(
                f'find "{target.models_dir}" -maxdepth 1 -name "*.gguf" -printf "%f|%s\\n" 2>/dev/null '
                f'|| stat -f "%N|%z" "{target.models_dir}"/*.gguf 2>/dev/null',
                timeout=15)
        files = []
        if result.ok and result.stdout:
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit("|", 1)
                if len(parts) == 2:
                    fname = parts[0].split("\\")[-1].split("/")[-1]
                    try:
                        size_bytes = int(parts[1])
                    except ValueError:
                        size_bytes = 0
                else:
                    fname = line.split("\\")[-1].split("/")[-1]
                    size_bytes = 0
                files.append((fname, size_bytes))
        # 与目录匹配，标注完整性
        catalog_names = {m["filename"]: m for m in list_all()}
        models = []
        for fname, size_bytes in files:
            entry = catalog_names.get(fname)
            expected_gb = entry["size_gb"] if entry else None
            size_gb = round(size_bytes / (1024 ** 3), 2)
            # 判断是否完整：有目录预期大小时，实际 >= 预期的 90% 视为完整
            complete = True
            if expected_gb and expected_gb > 0:
                complete = size_gb >= expected_gb * 0.9
            models.append({
                "filename": fname,
                "size_bytes": size_bytes,
                "size_gb": size_gb,
                "expected_gb": expected_gb,
                "complete": complete,
                "in_catalog": bool(entry),
                "model_id": entry["id"] if entry else None,
            })
        return {"models": models, "count": len(models)}
    finally:
        executor.close()
