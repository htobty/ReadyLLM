"""长视频编排服务（长视频第 3、4、5 层合一）

把一个分镜脚本（video_storyboard 产出）变成一部分钟级长视频。核心思路：
H3 单次只能生成 5-15 秒，长视频必须「多段串行生成 + 首尾帧衔接 + 拼接」。

执行流（后台线程，逐段串行，因显存只有一份不能并发）：
  段 0：用「种子首帧图」（助手生成或用户上传）走 I2V → 得 shot_0.mp4
  段 i>0：在目标机用 ffmpeg 抽 shot_(i-1) 的末帧到 input/ → 作为本段首帧走
          I2V → 得 shot_i.mp4（画面因此与上一段无缝衔接）
  全部完成：目标机 ffmpeg concat 所有段 → final.mp4

设计要点：
  - 任务状态落盘 ~/.model-deploy-assistant/long_video/，进程重启/前端刷新可恢复，
    已完成段不重跑（断点续跑）。
  - 抽帧、拼接全部在目标机本地用 ffmpeg 完成（已验证 ffmpeg 在 PATH），
    不把中间帧拉回控制端再传回去，省流量省时间。
  - 不硬编码任何个人环境：目标机 / 端口 / 目录全部来自 Target 配置。
  - 单段失败：记录该段 error，任务置为 failed，但保留已完成段，可修复后续跑。
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

_TASK_DIR = os.path.expanduser("~/.model-deploy-assistant/long_video")

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
                for s in job.get("shots", []):
                    if s.get("state") == "running":
                        s["state"] = "interrupted"
            _JOBS[job["job_id"]] = job
        except Exception:
            continue


def get_job(job_id: str) -> Optional[dict]:
    return _JOBS.get(job_id)


def start_long_video(target_id: str, storyboard: dict,
                     ref_image_paths: Optional[List[str]] = None,
                     width: int = 832, height: int = 480,
                     steps: int = 8, cfg: float = 1.0,
                     fps: int = 16) -> Dict[str, Any]:
    """提交一个长视频生成任务，立即返回 job_id，后台线程串行推进。

    ref_image_paths：控制端（本机）上的角色参考图绝对路径列表。每段都走
    R2V（MiniMaxH3ReferenceToVideo），用同一组参考图锁住人物身份，镜头之间
    正常硬切——不再用"上一段末帧当下一段首帧"的链式衔接（末帧已漂移，逐段
    累积会导致人物越往后越不像）。参考图只上传一次，所有段复用。
    """
    ref_image_paths = ref_image_paths or []
    shots = storyboard.get("shots") or []
    if not shots:
        return {"success": False, "message": "分镜脚本为空"}
    target = get_target(target_id)
    if not target:
        return {"success": False, "message": "目标机器不存在"}

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "target_id": target_id,
        "title": storyboard.get("title", ""),
        "status": "running",
        "width": width, "height": height,
        "steps": steps, "cfg": cfg, "fps": fps,
        "ref_image_paths": ref_image_paths,
        "shots": [
            {"index": s["index"], "title": s.get("title", ""),
             "prompt": s["prompt"], "length": s["length"],
             "state": "pending", "prompt_id": "", "output_file": "",
             "error": ""}
            for s in shots
        ],
        "final_file": "",
        "error": "",
        "created_at": time.time(),
    }
    with _LOCK:
        _JOBS[job_id] = job
    _save(job)

    t = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    t.start()
    return {"success": True, "job_id": job_id, "shot_count": len(job["shots"])}


def _comfy_input_dir(target) -> str:
    """目标机 ComfyUI 的 input 目录。"""
    base = target.engine_path or ""
    return path_join(target, base, "input")


def _comfy_output_dir(target) -> str:
    base = target.engine_path or ""
    return path_join(target, base, "output", "modeldeploy")


def _run_pipeline(job_id: str):
    """后台线程主逻辑：逐段串行生成 → 抽末帧衔接 → 拼接。"""
    job = _JOBS.get(job_id)
    if not job:
        return
    target = get_target(job["target_id"])
    if not target:
        job["status"] = "failed"; job["error"] = "目标机不存在"; _save(job); return

    executor = make_executor(target)
    try:
        engine = get_adapter(executor, target)
        if not hasattr(engine, "submit_workflow"):
            job["status"] = "failed"; job["error"] = "引擎不支持视频生成"; _save(job); return
        if not engine.is_running():
            job["status"] = "failed"; job["error"] = "ComfyUI 未运行，请先启动"; _save(job); return

        input_dir = _comfy_input_dir(target)
        # R2V：参考图只上传一次，所有段复用同一组，身份由模型内部对齐。
        ref_image_names = _upload_refs(executor, target, job, input_dir)
        if not ref_image_names:
            job["status"] = "failed"
            job["error"] = "角色参考图上传失败，R2V 无法锁身份"
            _save(job)
            return

        for shot in job["shots"]:
            if shot["state"] == "completed":
                continue  # 断点续跑：已完成段跳过。R2V 每段独立带参考图，无需末帧衔接
            ok = _generate_one_shot(engine, executor, target, job, shot, ref_image_names, input_dir)
            # aimdo 的 read_file_slice failed 是间歇性 bug（同参数上轮单段能成功），
            # 对失败段做最多 3 次重试吸收偶发；仍失败才判整段失败。
            attempt = 1
            while not ok and attempt < 3:
                attempt += 1
                shot["error"] = ""
                _save(job)
                time.sleep(3)
                ok = _generate_one_shot(engine, executor, target, job, shot, ref_image_names, input_dir)
            if not ok:
                job["status"] = "failed"
                job["error"] = f"第 {shot['index']} 段生成失败: {shot['error']}"
                _save(job)
                return
            # R2V 镜头之间正常硬切，不再抽末帧作下段首帧（末帧已漂移，链式会累积）

        # 全部段完成 → 拼接
        final = _concat_shots(executor, target, job, input_dir)
        if final:
            job["final_file"] = final
            job["status"] = "completed"
        else:
            job["status"] = "failed"
            job["error"] = "拼接成片失败"
        _save(job)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"任务异常: {e}"
        _save(job)
    finally:
        executor.close()


def _upload_refs(executor, target, job, input_dir) -> List[str]:
    """把控制端角色参考图上传到目标机 ComfyUI/input，返回文件名列表（只传一次，
    所有段复用）。R2V 用同一组参考图锁人物身份，镜头间硬切。任一图上传失败则
    返回已成功的列表；全失败返回空列表（调用方据此判任务失败）。"""
    paths = job.get("ref_image_paths") or []
    names: List[str] = []
    for i, p in enumerate(paths):
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                data = f.read()
        except Exception:
            continue
        ext = os.path.splitext(p)[1] or ".png"
        name = f"mdref_{job['job_id']}_{i}{ext}"
        remote = path_join(target, input_dir, name)
        if executor.write_file_bytes(data, remote):
            names.append(name)
    return names


def _files_from_outputs(outputs: Dict[str, Any]) -> List[dict]:
    """把 ComfyUI history 的 outputs（{node_id:{videos/gifs/images:[...]}}）
    解析成扁平文件列表。engine.get_progress 返回的是 outputs，不是路由层的 files。"""
    files: List[dict] = []
    for node_out in (outputs or {}).values():
        for key in ("gifs", "videos", "images"):
            for f in node_out.get(key, []) or []:
                files.append({
                    "filename": f.get("filename", ""),
                    "subfolder": f.get("subfolder", ""),
                    "type": f.get("type", "output"),
                })
    return files


def _restart_comfyui(engine, timeout=150) -> bool:
    """每段生成前干净重启 ComfyUI。

    上游 comfy_aimdo 的 read_file_slice failed 在模型被换出显存后冷加载时必现，
    且段数多 / 帧数大时连采样器读 UNet 权重都会崩（实测 9 段×103 帧段 2 即挂，
    节点 SamplerCustomAdvanced）。--disable-smart-memory 只能延缓不能根治。
    唯一可靠路径：保证每段都是干净进程的首个任务——生成前 stop→start→等待就绪。
    代价是每段重吃一次模型冷加载（约 30-60s）。
    """
    from ..services.engine_adapter import StartParams
    try:
        engine.stop()
    except Exception:
        pass
    time.sleep(8)  # 等端口释放
    try:
        engine.start(StartParams(model_path="", extra_args=[]))
    except Exception:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(6)
        try:
            if engine.is_running():
                return True
        except Exception:
            pass
    return False


def _generate_one_shot(engine, executor, target, job, shot,
                       ref_image_names, input_dir) -> bool:
    """生成单段（R2V）：干净重启 ComfyUI → 提交带参考图的 workflow → 轮询 → 记录产物。

    每段都用同一组 ref_image_names 走 R2V，身份由模型内部对齐，镜头间硬切。
    """
    shot["state"] = "running"
    _save(job)
    try:
        if not _restart_comfyui(engine):
            shot["state"] = "failed"; shot["error"] = "重启 ComfyUI 超时未就绪"
            _save(job); return False
        workflow = engine.build_video_workflow(
            prompt=shot["prompt"],
            model_name="",
            width=job["width"], height=job["height"],
            length=shot["length"],
            steps=job["steps"], cfg=job["cfg"],
            fps=job["fps"],
            ref_image_names=ref_image_names,
        )
        ok, result = engine.submit_workflow(workflow)
        if not ok:
            shot["state"] = "failed"; shot["error"] = str(result); _save(job); return False
        shot["prompt_id"] = result
        _save(job)

        # 轮询直到完成（单段上限 ~6 分钟）
        deadline = time.time() + 360
        while time.time() < deadline:
            time.sleep(8)
            prog = engine.get_progress(result)
            st = prog.get("state")
            if st == "completed":
                files = _files_from_outputs(prog.get("outputs"))
                mp4 = [f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".gif"))]
                if mp4:
                    shot["output_file"] = mp4[0]["filename"]
                    shot["state"] = "completed"
                    _save(job)
                    return True
                shot["state"] = "failed"; shot["error"] = "完成但无产物文件"; _save(job); return False
            if st == "error":
                shot["state"] = "failed"; shot["error"] = "ComfyUI 执行报错"; _save(job); return False
        shot["state"] = "failed"; shot["error"] = "生成超时"; _save(job); return False
    except Exception as e:
        shot["state"] = "failed"; shot["error"] = str(e); _save(job); return False


def _extract_last_frame(executor, target, output_file, input_dir) -> str:
    """在目标机用 ffmpeg 抽某段末帧到 input，返回帧图文件名（作下一段首帧）。失败返回空串。"""
    if not output_file:
        return ""
    out_mp4 = path_join(target, _comfy_output_dir(target), output_file)
    frame_name = f"mdframe_{uuid.uuid4().hex[:8]}.png"
    frame_path = path_join(target, input_dir, frame_name)
    # -sseof -0.1 定位到末尾 0.1s，-update 1 -vframes 1 只出一帧
    cmd = (f'ffmpeg -y -sseof -0.1 -i "{out_mp4}" -update 1 -vframes 1 '
           f'"{frame_path}"')
    r = executor.run(cmd, timeout=60)
    if r.ok:
        return frame_name
    return ""


def _concat_shots(executor, target, job, input_dir) -> str:
    """目标机 ffmpeg concat 所有段为 final.mp4，返回文件名。失败返回空串。"""
    out_dir = _comfy_output_dir(target)
    files = [s["output_file"] for s in job["shots"] if s.get("output_file")]
    if not files:
        return ""
    # 写 concat 列表文件（每行 file '绝对路径'）
    list_lines = "\n".join(f"file '{path_join(target, out_dir, f)}'" for f in files)
    list_name = f"mdconcat_{job['job_id']}.txt"
    list_path = path_join(target, out_dir, list_name)
    if not executor.write_file_bytes(list_lines.encode("utf-8"), list_path):
        return ""
    final_name = f"mdfinal_{job['job_id']}.mp4"
    final_path = path_join(target, out_dir, final_name)
    # -c copy 要求所有段编码参数一致（同 H3 配置生成满足）
    cmd = (f'ffmpeg -y -f concat -safe 0 -i "{list_path}" -c copy "{final_path}"')
    r = executor.run(cmd, timeout=120)
    if r.ok:
        return final_name
    # copy 失败回退重编码
    cmd2 = (f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-c:v libx264 -pix_fmt yuv420p "{final_path}"')
    r2 = executor.run(cmd2, timeout=240)
    return final_name if r2.ok else ""
