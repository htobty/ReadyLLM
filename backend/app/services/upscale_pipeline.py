"""成片超分服务：把一段（或拼接后的）低分辨率成片整体超到 1080P。

单图超分端点 /upscale 只能处理一张图；视频要「抽帧 → 逐帧超分 → 重新合成 →
接回原音轨」。本模块把这条批处理链封装成后台任务，状态落盘可查进度、可断点续跑。

执行流（后台线程，逐帧串行，因显存只有一份不能并发）：
  1. 目标机 ffmpeg 把成片所有帧抽成 frame_%05d.png
  2. 逐帧提交 build_upscale_workflow（RealESRGAN_x4plus → lanczos 收敛到
     out_w×out_h），每帧用唯一 filename_prefix（含帧号）保证输出可按序还原
  3. 目标机 ffmpeg 把超分后的帧序列按原 fps 合成无声视频
  4. 把原成片的音轨 mux 回超分视频（-c:v copy 不重编码，-map 1:a? 容错无音轨）

设计要点（与 video_pipeline 一致）：
  - 抽帧、合成、混音全部在目标机本地 ffmpeg 完成，不拉回控制端，省流量。
  - 不硬编码任何个人环境：目标机 / 端口 / 目录全部来自 Target 配置。
  - 任务状态落盘 ~/.model-deploy-assistant/upscale_video/，重启可恢复。
  - 逐帧超分记录已完成的帧号，断点续跑时跳过。
"""

import json
import os
import threading
import time
import uuid
from typing import Optional, Dict, Any, List

from ..models.target import get_target
from .executor import make_executor
from .engine_registry import get_adapter
from .collectors import path_join

_TASK_DIR = os.path.expanduser("~/.model-deploy-assistant/upscale_video")

# 内存注册表：job_id -> job dict（同时落盘，重启后从盘加载）
_JOBS: Dict[str, dict] = {}
_LOCK = threading.Lock()


def _task_path(job_id: str) -> str:
    return os.path.join(_TASK_DIR, f"{job_id}.json")


def _save(job: dict):
    os.makedirs(_TASK_DIR, exist_ok=True)
    with open(_task_path(job["job_id"]), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def load_all_jobs():
    """进程启动时从盘恢复历史任务（运行中的任务因线程已死标记为 interrupted）。"""
    if not os.path.isdir(_TASK_DIR):
        return
    for name in os.listdir(_TASK_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(_TASK_DIR, name), "r", encoding="utf-8") as f:
                job = json.load(f)
            if job.get("status") == "running":
                job["status"] = "interrupted"
            _JOBS[job["job_id"]] = job
        except Exception:
            continue


def get_job(job_id: str) -> Optional[dict]:
    return _JOBS.get(job_id)


def _comfy_input_dir(target) -> str:
    base = target.engine_path or ""
    return path_join(target, base, "input")


def _comfy_output_dir(target) -> str:
    base = target.engine_path or ""
    return path_join(target, base, "output", "modeldeploy")


def start_upscale_video(target_id: str, filename: str,
                        subfolder: str = "modeldeploy",
                        out_w: int = 1920, out_h: int = 1080,
                        fps: int = 16) -> Dict[str, Any]:
    """提交一个成片超分任务，立即返回 job_id，后台线程逐帧推进。

    filename：ComfyUI output 里的成片文件名（如 mdfinal_xxx.mp4）。
    subfolder：相对 output 的子目录，默认 modeldeploy。
    out_w/out_h：目标分辨率（默认 1920×1080）。
    fps：合成帧率，应与源片一致（H3 长视频常用 16）。
    """
    target = get_target(target_id)
    if not target:
        return {"success": False, "message": "目标机器不存在"}

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "target_id": target_id,
        "status": "running",
        "src_file": filename,
        "subfolder": subfolder,
        "out_w": out_w, "out_h": out_h, "fps": fps,
        "total_frames": 0,
        "done_frames": 0,
        "done_indices": [],
        "final_file": "",
        "error": "",
        "created_at": time.time(),
    }
    with _LOCK:
        _JOBS[job_id] = job
    _save(job)

    t = threading.Thread(target=_run_upscale, args=(job_id,), daemon=True)
    t.start()
    return {"success": True, "job_id": job_id}


def _run_upscale(job_id: str):
    """后台线程主逻辑：抽帧 → 逐帧超分 → 合成 → 接音轨。"""
    job = _JOBS.get(job_id)
    if not job:
        return
    target = get_target(job["target_id"])
    if not target:
        job["status"] = "failed"; job["error"] = "目标机不存在"; _save(job); return

    executor = make_executor(target)
    try:
        engine = get_adapter(executor, target)
        if not hasattr(engine, "build_upscale_workflow"):
            job["status"] = "failed"; job["error"] = "引擎不支持超分"; _save(job); return
        if not engine.is_running():
            job["status"] = "failed"; job["error"] = "ComfyUI 未运行，请先启动"; _save(job); return

        out_dir = _comfy_output_dir(target)
        input_dir = _comfy_input_dir(target)
        src_path = path_join(target, out_dir if job["subfolder"] == "modeldeploy"
                             else path_join(target, target.engine_path or "", "output", job["subfolder"]),
                             job["src_file"])
        # 帧工作目录（目标机 output/modeldeploy 下）
        frames_dir = path_join(target, out_dir, f"frames_{job_id}")
        ups_dir = path_join(target, out_dir, f"ups_{job_id}")
        executor.run(f'mkdir "{frames_dir}"', timeout=30)
        executor.run(f'mkdir "{ups_dir}"', timeout=30)

        # 1. 抽帧：frame_%05d.png
        job["status"] = "extracting"; _save(job)
        r = executor.run(f'ffmpeg -y -i "{src_path}" "{path_join(target, frames_dir, "frame_%05d.png")}"',
                         timeout=180)
        if not r.ok:
            job["status"] = "failed"; job["error"] = "抽帧失败: " + (r.stderr or "")[:200]
            _save(job); return
        # 列出帧数
        lr = executor.run(f'dir /b "{path_join(target, frames_dir, "frame_*.png")}"', timeout=30)
        frame_names = [x.strip() for x in (lr.stdout or "").splitlines()
                       if x.strip().lower().endswith(".png")]
        frame_names.sort()
        job["total_frames"] = len(frame_names)
        _save(job)
        if not frame_names:
            job["status"] = "failed"; job["error"] = "抽帧结果为空"; _save(job); return

        # 2. 逐帧超分：每帧唯一 prefix 保序，输出 ups_%05d_00001_.png
        job["status"] = "upscaling"; _save(job)
        done = set(job.get("done_indices") or [])
        for idx, fname in enumerate(frame_names):
            if idx in done:
                continue
            # 把帧从 frames_dir 拷到 input（LoadImage 只认 input 目录）
            src_frame = path_join(target, frames_dir, fname)
            in_name = f"mdupf_{job_id}_{idx:05d}.png"
            cp = executor.run(f'copy /Y "{src_frame}" "{path_join(target, input_dir, in_name)}"',
                              timeout=30)
            if not cp.ok:
                job["status"] = "failed"
                job["error"] = f"帧 {idx} 拷贝到 input 失败"; _save(job); return
            wf = engine.build_upscale_workflow(
                image_name=in_name, out_w=job["out_w"], out_h=job["out_h"],
                filename_prefix=f"modeldeploy/ups_{job_id}/u{idx:05d}")
            ok, pid = engine.submit_workflow(wf)
            if not ok:
                job["status"] = "failed"; job["error"] = f"帧 {idx} 提交失败: {pid}"
                _save(job); return
            pr = _wait_frame(engine, pid, timeout=120)
            if pr != "completed":
                job["status"] = "failed"
                job["error"] = f"帧 {idx} 超分{pr}"; _save(job); return
            done.add(idx)
            job["done_indices"] = sorted(done)
            job["done_frames"] = len(done)
            _save(job)

        # 3. 合成无声视频：image2 序列按帧号 → 超分视频
        job["status"] = "compositing"; _save(job)
        # 超分输出在 output/modeldeploy/ups_<job>/uNNNNN_00001_.png，需重命名对齐
        # 用 ffmpeg 的 pattern 不便（ComfyUI 加了 _00001_ 后缀），改用 concat 列表。
        ups_dir_abs = path_join(target, out_dir, f"ups_{job_id}")
        lu = executor.run(f'dir /b "{path_join(target, ups_dir_abs, "u*.png")}"', timeout=30)
        ups_files = sorted(x.strip() for x in (lu.stdout or "").splitlines()
                           if x.strip().lower().endswith(".png"))
        if len(ups_files) != len(frame_names):
            job["status"] = "failed"
            job["error"] = f"超分帧数({len(ups_files)})≠源帧数({len(frame_names)})"
            _save(job); return
        # 写 image2 符号链接不便，用 concat demuxer 逐帧（每帧 duration=1/fps）
        list_name = f"mdupsconcat_{job_id}.txt"
        list_path = path_join(target, out_dir, list_name)
        dur = round(1.0 / max(1, job["fps"]), 6)
        lines = []
        for uf in ups_files:
            lines.append(f"file '{path_join(target, ups_dir_abs, uf)}'")
            lines.append(f"duration {dur}")
        lines.append(f"file '{path_join(target, ups_dir_abs, ups_files[-1])}'")
        if not executor.write_file_bytes("\n".join(lines).encode("utf-8"), list_path):
            job["status"] = "failed"; job["error"] = "写合成列表失败"; _save(job); return
        silent_name = f"mdupssilent_{job_id}.mp4"
        silent_path = path_join(target, out_dir, silent_name)
        cr = executor.run(
            f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-vf "fps={job["fps"]}" '
            f'-c:v libx264 -pix_fmt yuv420p "{silent_path}"', timeout=300)
        if not cr.ok:
            job["status"] = "failed"; job["error"] = "合成失败: " + (cr.stderr or "")[:200]
            _save(job); return

        # 4. 接回原音轨：超分视频 + 原片音轨（-map 1:a? 容错无音轨）
        final_name = f"mdup1080_{job_id}.mp4"
        final_path = path_join(target, out_dir, final_name)
        mr = executor.run(
            f'ffmpeg -y -i "{silent_path}" -i "{src_path}" '
            f'-map 0:v -map 1:a? -c:v copy -c:a aac -shortest "{final_path}"',
            timeout=180)
        if not mr.ok:
            job["status"] = "failed"; job["error"] = "混音失败: " + (mr.stderr or "")[:200]
            _save(job); return

        job["final_file"] = final_name
        job["status"] = "completed"
        _save(job)
    except Exception as e:
        job["status"] = "failed"; job["error"] = f"任务异常: {e}"; _save(job)
    finally:
        executor.close()


def _wait_frame(engine, prompt_id: str, timeout: int = 120) -> str:
    """轮询单帧超分任务，返回 completed/error/timeout。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            prog = engine.get_progress(prompt_id)
        except Exception:
            continue
        st = prog.get("state")
        if st == "completed":
            return "completed"
        if st == "error":
            return "error"
    return "timeout"
