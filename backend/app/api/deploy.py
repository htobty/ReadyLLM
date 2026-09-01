"""部署管理 API（基于用户配置的 Target）"""

import os
import json
import shlex

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

from ..models.target import get_target
from ..services.executor import make_executor
from ..services.engine_adapter import StartParams
from ..services.engine_registry import get_adapter
from ..services.collectors import path_join, detect_hardware
from ..services import tune_history

router = APIRouter()


class DeployRequest(BaseModel):
    target_id: str
    model: str
    # 用户手动编辑的命令行参数文本（优先）；如 "--ctx-size 8192 --batch-size 4096"
    args_text: Optional[str] = None
    # 向后兼容：直接传参数列表
    extra_args: Optional[list[str]] = None


def _adapter(target_id: str):
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在，请先在设置中配置")
    executor = make_executor(target)
    return target, executor, get_adapter(executor, target)


# ==================== 运行中模型记录 ====================
# 记录每个 target 当前正在运行的模型名，供前端刷新后固定选中（不再回退默认）。
# 持久化到本地 JSON，后端重启不丢失。
_RUNNING_FILE = os.path.expanduser("~/.model-deploy-assistant/running_models.json")


def _load_running() -> dict:
    try:
        with open(_RUNNING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return {}


def _save_running(data: dict):
    os.makedirs(os.path.dirname(_RUNNING_FILE), exist_ok=True)
    with open(_RUNNING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _record_running(target_id: str, model: str):
    data = _load_running()
    data[target_id] = model
    _save_running(data)


def _clear_running(target_id: str):
    data = _load_running()
    if target_id in data:
        del data[target_id]
        _save_running(data)


def _get_running(target_id: str) -> str:
    return _load_running().get(target_id, "")


@router.get("/models")
def list_models(target_id: str):
    """列出目标机器模型目录下的 .gguf 文件"""
    target, executor, _ = _adapter(target_id)
    try:
        if not target.models_dir:
            return {"models": [], "count": 0, "error": "未配置模型目录"}
        if target.os == "windows":
            pattern = f'{target.models_dir}\\*.gguf'
            result = executor.run(f'dir /b "{pattern}"', timeout=10)
        else:
            result = executor.run(f'ls -1 "{target.models_dir}"/*.gguf 2>/dev/null', timeout=10)
        models = []
        if result.ok and result.stdout:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    models.append(line.split("\\")[-1].split("/")[-1])
        return {"models": sorted(models), "count": len(models)}
    finally:
        executor.close()


@router.get("/video-models")
def list_video_models(target_id: str):
    """扫描目标机 ComfyUI 的 diffusion_models 目录，列出真实存在的视频模型。

    与文本部署扫 .gguf 同理：只列机器上实际下载好的权重，不再用写死清单。
    ComfyUI 约定 diffusion_models 子目录存放主扩散模型（UNet/DiT）。
    """
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在")
    if not target.models_dir:
        return {"models": [], "count": 0, "error": "未配置模型目录"}

    diff_dir = path_join(target, target.models_dir, "diffusion_models")
    executor = make_executor(target)
    try:
        if target.os == "windows":
            pattern = f'{diff_dir}\\*.safetensors'
            result = executor.run(f'dir /b "{pattern}"', timeout=10)
        else:
            result = executor.run(f'ls -1 "{diff_dir}"/*.safetensors 2>/dev/null', timeout=10)
        models = []
        if result.ok and result.stdout:
            for line in result.stdout.splitlines():
                fn = line.strip().split("\\")[-1].split("/")[-1]
                # 过滤 ComfyUI 占位文件（put_xxx_here）与空名
                if not fn or fn.lower().startswith("put_"):
                    continue
                models.append({
                    "filename": fn,
                    "name": _pretty_video_name(fn),
                })
        return {"models": sorted(models, key=lambda m: m["filename"]), "count": len(models)}
    finally:
        executor.close()


def _pretty_video_name(filename: str) -> str:
    """把权重文件名美化为可读标签（不依赖任何硬编码个人环境）。"""
    low = filename.lower()
    if "minimax" in low and "h3" in low:
        tag = "pruned int8" if "pruned" in low and "int8" in low else (
            "fp8" if "fp8" in low else ("bf16" if "bf16" in low else "int8"))
        return f"MiniMax H3 · {tag}"
    if "wan" in low:
        return "Wan 2.1"
    if "ltx" in low:
        return "LTX-Video"
    if "cogvideo" in low:
        return "CogVideoX"
    return filename.rsplit(".", 1)[0]


# ComfyUI output 根目录：优先 engine_path/output，回退 models_dir 同级 output
def _comfy_output_root(target) -> str:
    base = target.engine_path or (target.models_dir or "").rstrip("\\/").rsplit("\\/", 1)[0].rsplit("/", 1)[0]
    if not base:
        return ""
    return path_join(target, base, "output")


def _safe_join(target, root: str, *parts: str) -> Optional[str]:
    """把若干路径片段安全拼到 root 下，拒绝目录穿越（含 .. 或绝对路径）。"""
    p = root
    for seg in parts:
        if seg is None:
            continue
        s = seg.strip()
        if not s:
            continue
        if ".." in s or s.startswith("/") or s.startswith("\\") or ":" in s:
            return None
        p = path_join(target, p, s)
    return p


@router.get("/video-file")
def get_video_file(target_id: str, filename: str, subfolder: str = ""):
    """经 SSH/本地读取目标机 ComfyUI output 下的成片，以视频字节流返回，
    供前端 <video> 直接预览（控制端无法直连目标机 ComfyUI 端口时的代理）。"""
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在")
    root = _comfy_output_root(target)
    if not root:
        raise HTTPException(status_code=400, detail="未配置 ComfyUI 目录")
    full = _safe_join(target, root, subfolder, filename)
    if not full:
        raise HTTPException(status_code=400, detail="非法文件路径")

    executor = make_executor(target)
    try:
        data = executor.read_file_bytes(full)
    finally:
        executor.close()
    if data is None:
        raise HTTPException(status_code=404, detail="成片文件不存在或读取失败")

    low = filename.lower()
    media = "video/webm" if low.endswith(".webm") else "video/mp4"
    return Response(content=data, media_type=media,
                    headers={"Cache-Control": "no-store"})


@router.post("/start")
def start_model(req: DeployRequest):
    """启动模型（支持用户手动填写的运行参数）"""
    target, executor, engine = _adapter(req.target_id)
    try:
        model_path = path_join(target, target.models_dir, req.model)
        # 解析参数：args_text 优先，回退 extra_args
        if req.args_text and req.args_text.strip():
            try:
                extra = shlex.split(req.args_text.strip())
            except ValueError as e:
                return {"success": False, "message": f"参数格式错误: {e}"}
        else:
            extra = list(req.extra_args or [])
        # 监控依赖 metrics；服务需对外可达 host。缺失则补，重复则去
        joined = " ".join(extra)
        if "--metrics" not in joined:
            extra.append("--metrics")
        if "--host" not in joined:
            extra += ["--host", "0.0.0.0"]
        params = StartParams(model_path=model_path, extra_args=extra)
        success, msg = engine.start(params)
        if success:
            _record_running(req.target_id, req.model)
        return {"success": success, "message": msg, "args": extra}
    finally:
        executor.close()


@router.post("/stop")
def stop_model(target_id: str):
    """停止模型"""
    target, executor, engine = _adapter(target_id)
    try:
        success, msg = engine.stop()
        if success:
            _clear_running(target_id)
        return {"success": success, "message": msg}
    finally:
        executor.close()


@router.get("/status")
def get_status(target_id: str):
    """获取运行状态。model 字段返回当前运行中的模型名（若有），
    供前端刷新后固定选中正在运行的模型，而非回退到列表第一个。"""
    target, executor, engine = _adapter(target_id)
    try:
        running = engine.is_running()
        model = _get_running(target_id) if running else ""
        return {"running": running, "engine": engine.name(), "model": model}
    finally:
        executor.close()


# ==================== 视频生成（ComfyUI） ====================

class VideoGenerateRequest(BaseModel):
    target_id: str
    prompt: str
    model_name: str
    negative_prompt: Optional[str] = ""
    width: int = 832
    height: int = 480
    length: int = 49
    steps: int = 30
    cfg: float = 6.0
    seed: Optional[int] = None
    fps: int = 16
    # 是否让大模型把粗略描述编排成电影级提示词并推荐采样参数（路线 A）
    enhance: bool = False
    # I2V：控制端（本机）上的首帧图绝对路径。非空则上传到目标机 ComfyUI/input
    # 并走图生视频；为空则纯 T2V。
    image_path: Optional[str] = None
    # R2V：控制端上的多张角色参考图绝对路径列表。非空则走参考图生视频
    # （MiniMaxH3ReferenceToVideo，身份由模型内部对齐，锁人物一致性）。
    # 优先级高于 image_path。
    ref_image_paths: Optional[list] = None
    # TeaCache 采样加速：跳过相邻冗余去噪步。实测约 1.3-3× 提速（视步数）。
    teacache: bool = False
    teacache_thresh: float = 0.15


@router.post("/generate")
def generate_video(req: VideoGenerateRequest):
    """向 ComfyUI 提交一次 text-to-video 生成任务，返回 prompt_id。

    仅适用于 engine_type=comfyui 的 Target。提交后由前端轮询
    /generate/progress 获取状态与成片路径。"""
    target, executor, engine = _adapter(req.target_id)
    try:
        if not hasattr(engine, "submit_workflow"):
            return {"success": False, "message": "当前目标机引擎不支持视频生成，请改用 ComfyUI"}
        if not engine.is_running():
            return {"success": False, "message": "ComfyUI 服务未运行，请先在部署页启动"}

        # 路线 A：可选的 LLM 提示词编排。失败优雅降级为原始 prompt，绝不阻断生成。
        prompt = req.prompt
        steps = req.steps
        cfg = req.cfg
        enhanced = False
        reasoning = ""
        if req.enhance:
            from ..services.video_prompt import enhance_prompt
            # 按输入推断 H3 生成模式：有参考图走 R2V（六段式锁身份），
            # 有首帧图走 I2V（三字段+对齐指令），否则 T2V。
            if req.ref_image_paths:
                _mode, _pc = "r2v", len(req.ref_image_paths)
            elif req.image_path:
                _mode, _pc = "i2v", 1
            else:
                _mode, _pc = "t2v", 0
            try:
                e = enhance_prompt(req.prompt, mode=_mode, picture_count=_pc)
            except Exception:
                e = None
            if e:
                prompt = e["prompt"]
                steps = e["steps"]
                cfg = e["cfg"]
                reasoning = e.get("reasoning", "")
                enhanced = True

        # I2V：把控制端本地首帧图上传到目标机 ComfyUI/input，取回文件名
        image_name = ""
        upload_err = ""
        if req.image_path:
            import os as _os
            import uuid as _uuid
            if not _os.path.exists(req.image_path):
                return {"success": False, "message": f"首帧图不存在: {req.image_path}"}
            try:
                with open(req.image_path, "rb") as _f:
                    img_bytes = _f.read()
            except Exception as e:
                return {"success": False, "message": f"读取首帧图失败: {e}"}
            ext = _os.path.splitext(req.image_path)[1] or ".png"
            image_name = f"mdframe_{_uuid.uuid4().hex[:8]}{ext}"
            input_dir = path_join(target, target.engine_path or "", "input")
            remote = path_join(target, input_dir, image_name)
            if not executor.write_file_bytes(img_bytes, remote):
                return {"success": False, "message": "首帧图上传到目标机 input 目录失败"}

        # R2V：把控制端多张角色参考图上传到目标机 ComfyUI/input，收集文件名列表。
        # 优先级高于 I2V（同一请求两者都给时走 R2V）。
        ref_image_names = []
        if req.ref_image_paths:
            import os as _os
            import uuid as _uuid
            input_dir = path_join(target, target.engine_path or "", "input")
            for rp in req.ref_image_paths:
                if not _os.path.exists(rp):
                    return {"success": False, "message": f"参考图不存在: {rp}"}
                try:
                    with open(rp, "rb") as _f:
                        rb = _f.read()
                except Exception as e:
                    return {"success": False, "message": f"读取参考图失败: {e}"}
                ext = _os.path.splitext(rp)[1] or ".png"
                rname = f"mdref_{_uuid.uuid4().hex[:8]}{ext}"
                rremote = path_join(target, input_dir, rname)
                if not executor.write_file_bytes(rb, rremote):
                    return {"success": False, "message": f"参考图上传失败: {rp}"}
                ref_image_names.append(rname)

        workflow = engine.build_video_workflow(
            prompt=prompt,
            model_name=req.model_name,
            negative_prompt=req.negative_prompt or "",
            width=req.width,
            height=req.height,
            length=req.length,
            steps=steps,
            cfg=cfg,
            seed=req.seed,
            fps=req.fps,
            image_name=image_name,
            ref_image_names=ref_image_names,
            teacache=req.teacache,
            teacache_thresh=req.teacache_thresh,
        )
        ok, result = engine.submit_workflow(workflow)
        if not ok:
            return {"success": False, "message": result}
        return {
            "success": True,
            "prompt_id": result,
            "message": "生成任务已提交",
            "enhanced": enhanced,
            "i2v": bool(image_name) and not ref_image_names,
            "r2v": bool(ref_image_names),
            "final_prompt": prompt,
            "final_steps": steps,
            "final_cfg": cfg,
            "reasoning": reasoning,
        }
    finally:
        executor.close()


@router.get("/generate/progress")
def generate_progress(target_id: str, prompt_id: str):
    """查询视频生成任务状态；完成时返回成片信息。"""
    target, executor, engine = _adapter(target_id)
    try:
        if not hasattr(engine, "get_progress"):
            return {"state": "error", "message": "当前引擎不支持生成任务查询"}
        prog = engine.get_progress(prompt_id)
        if prog.get("state") == "completed":
            outputs = prog.get("outputs") or {}
            files = []
            for node_out in outputs.values():
                for key in ("gifs", "videos", "images"):
                    for f in node_out.get(key, []) or []:
                        files.append({
                            "filename": f.get("filename", ""),
                            "subfolder": f.get("subfolder", ""),
                            "type": f.get("type", "output"),
                        })
            return {"state": "completed", "files": files}
        return {"state": prog.get("state", "unknown")}
    finally:
        executor.close()



# ==================== 默认参数（调优回填 / 确定性回退） ====================

# 命令行展示顺序（影响观感，不影响功能）
_ARG_ORDER = [
    "ctx-size", "n-gpu-layers", "batch-size", "ubatch-size",
    "cache-type-k", "cache-type-v", "flash-attn", "fit",
    "spec-type", "spec-draft-n-max", "spec-draft-n-min",
    "gpu-layers-draft", "spec-draft-ngl", "threads",
]


def _params_to_args_str(params: dict) -> str:
    """扁平参数字典 -> 可编辑命令行字符串"""
    keys = [k for k in _ARG_ORDER if k in params] + \
           [k for k in params if k not in _ARG_ORDER]
    parts = []
    for k in keys:
        v = params.get(k, "")
        if v == "" or v is None:
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)


def _model_size_gb(executor, target, model: str) -> float:
    """查目标机上模型文件实际大小（GB），失败返回 0"""
    p = path_join(target, target.models_dir, model)
    if target.os == "windows":
        cmd = f'powershell -Command "(Get-Item \'{p}\').Length"'
    else:
        cmd = f'wc -c < "{p}"'
    r = executor.run(cmd, timeout=15)
    for tok in (r.stdout or "").split():
        if tok.isdigit():
            return round(int(tok) / (1024 ** 3), 2)
    return 0.0


@router.get("/default-args")
def default_args(target_id: str, model: str):
    """返回部署默认参数：优先最近调优结果，无则用确定性生成器现算。

    返回可直接编辑的命令行字符串，供前端预填到参数框。
    """
    target = get_target(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="目标机器不存在")

    # 1) 优先：最近一次调优参数
    rec = tune_history.get_latest(target_id, model)
    if rec and rec.get("params"):
        return {
            "args": _params_to_args_str(rec["params"]),
            "source": rec.get("source", "tuner"),
            "score": rec.get("score", 0),
            "ts": rec.get("ts", ""),
        }

    # 2) 回退：确定性生成器（需现采硬件 + 模型大小）
    executor = make_executor(target)
    try:
        from ..services.config_generator import generate_config
        hw = detect_hardware(executor, target)
        gpu = hw.get("gpu") or {}
        cpu = hw.get("cpu") or {}
        mem = hw.get("memory") or {}
        vram = gpu.get("total_memory_gb", 0) or 0
        if not vram and mem.get("total_gb"):
            # Apple Silicon 统一内存：按物理内存×0.75 估算可用显存
            vram = round(mem["total_gb"] * 0.75, 1)
        size_gb = _model_size_gb(executor, target, model)
        cores = cpu.get("cores", 8) or 8
        threads = cpu.get("threads", 16) or 16
        gen = generate_config(
            gpu_vram_gb=vram, model_size_gb=size_gb, model_filename=model,
            ctx_size=8192, cpu_cores=cores, cpu_threads=threads,
        )
        return {
            "args": _params_to_args_str(gen.get("params", {})),
            "source": "generated",
            "score": 0,
            "ts": "",
            "reasoning": gen.get("reasoning", []),
        }
    except Exception as e:
        # 3) 兜底：空参数，让后端用引擎默认
        return {"args": "", "source": "default", "score": 0, "ts": "", "error": str(e)}
    finally:
        executor.close()



# ==================== 长视频（分镜 → 逐段 I2V → 拼接） ====================

class StoryboardRequest(BaseModel):
    theme: str
    total_seconds: int = 60
    max_shots: int = 12


@router.post("/storyboard")
def make_storyboard(req: StoryboardRequest):
    """把主题拆成分镜脚本（不落库，仅返回预览供用户确认/编辑）。"""
    from ..services.video_storyboard import generate_storyboard
    sb = generate_storyboard(req.theme, req.total_seconds, req.max_shots)
    if not sb:
        return {"success": False,
                "message": "分镜生成失败：请确认已在设置中配置可用的大模型 API"}
    return {"success": True, "storyboard": sb}


class LongVideoRequest(BaseModel):
    target_id: str
    storyboard: dict
    # R2V：角色参考图绝对路径列表，每段共用同一组锁身份，镜头间硬切。
    ref_image_paths: Optional[list] = None
    width: int = 832
    height: int = 480
    steps: int = 8
    cfg: float = 1.0
    fps: int = 16


@router.post("/long-video")
def submit_long_video(req: LongVideoRequest):
    """提交长视频生成任务：按分镜逐段 R2V 串行生成（同一组参考图锁人物身份）、
    镜头硬切、拼接成片。立即返回 job_id，后台线程推进；前端轮询
    /long-video/progress 看逐段进度。"""
    from ..services.video_pipeline import start_long_video
    res = start_long_video(
        target_id=req.target_id,
        storyboard=req.storyboard,
        ref_image_paths=req.ref_image_paths or [],
        width=req.width, height=req.height,
        steps=req.steps, cfg=req.cfg, fps=req.fps,
    )
    return res


@router.get("/long-video/progress")
def long_video_progress(job_id: str):
    """查询长视频任务逐段进度；完成时返回拼接成片文件名（供 /video-file 预览）。"""
    from ..services.video_pipeline import get_job
    job = get_job(job_id)
    if not job:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "title": job.get("title", ""),
        "shots": [
            {"index": s["index"], "title": s.get("title", ""),
             "state": s["state"], "error": s.get("error", "")}
            for s in job["shots"]
        ],
        "final_file": job.get("final_file", ""),
        "target_id": job.get("target_id", ""),
        "error": job.get("error", ""),
    }



class UpscaleRequest(BaseModel):
    target_id: str
    # 二选一：image_path 是控制端本地图片绝对路径（会上传到目标机 input），
    # image_name 是已在 ComfyUI input 目录里的图名。
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    out_w: int = 1920
    out_h: int = 1080


@router.post("/upscale")
def upscale_image(req: UpscaleRequest):
    """提交单图 AI 超分（RealESRGAN_x4plus → lanczos 收敛到 out_w×out_h）。

    复用 ComfyUI 引擎；提交后由 /generate/progress 轮询（SaveImage 输出在
    images 字段，progress 逻辑已收集）。仅 engine_type=comfyui 的 Target 可用。"""
    target, executor, engine = _adapter(req.target_id)
    try:
        if not hasattr(engine, "build_upscale_workflow"):
            return {"success": False, "message": "当前引擎不支持超分，请改用 ComfyUI"}
        if not engine.is_running():
            return {"success": False, "message": "ComfyUI 服务未运行，请先在部署页启动"}
        name = req.image_name or ""
        if req.image_path:
            import os as _os
            import uuid as _uuid
            if not _os.path.exists(req.image_path):
                return {"success": False, "message": f"图片不存在: {req.image_path}"}
            try:
                with open(req.image_path, "rb") as _f:
                    b = _f.read()
            except Exception as e:
                return {"success": False, "message": f"读取图片失败: {e}"}
            ext = _os.path.splitext(req.image_path)[1] or ".png"
            name = f"mdup_{_uuid.uuid4().hex[:8]}{ext}"
            input_dir = path_join(target, target.engine_path or "", "input")
            if not executor.write_file_bytes(b, path_join(target, input_dir, name)):
                return {"success": False, "message": "图片上传到目标机 input 目录失败"}
        if not name:
            return {"success": False, "message": "需提供 image_path 或 image_name"}
        wf = engine.build_upscale_workflow(
            image_name=name, out_w=req.out_w, out_h=req.out_h)
        ok, result = engine.submit_workflow(wf)
        if not ok:
            return {"success": False, "message": result}
        return {"success": True, "prompt_id": result, "message": "超分任务已提交"}
    finally:
        executor.close()


class UpscaleVideoRequest(BaseModel):
    target_id: str
    # ComfyUI output 里的成片文件名（如 mdfinal_xxx.mp4）
    filename: str
    subfolder: str = "modeldeploy"
    out_w: int = 1920
    out_h: int = 1080
    fps: int = 16


@router.post("/upscale-video")
def submit_upscale_video(req: UpscaleVideoRequest):
    """提交成片整体超分任务：抽帧 → 逐帧 RealESRGAN 超分到 out_w×out_h →
    按原 fps 合成 → 接回原音轨。立即返回 job_id，后台线程推进；前端轮询
    /upscale-video/progress 看逐帧进度。"""
    from ..services.upscale_pipeline import start_upscale_video
    return start_upscale_video(
        target_id=req.target_id,
        filename=req.filename,
        subfolder=req.subfolder,
        out_w=req.out_w, out_h=req.out_h, fps=req.fps,
    )


@router.get("/upscale-video/progress")
def upscale_video_progress(job_id: str):
    """查询成片超分任务进度；完成时返回超分成片文件名（供 /video-file 预览）。"""
    from ..services.upscale_pipeline import get_job
    job = get_job(job_id)
    if not job:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "total_frames": job.get("total_frames", 0),
        "done_frames": job.get("done_frames", 0),
        "final_file": job.get("final_file", ""),
        "target_id": job.get("target_id", ""),
        "error": job.get("error", ""),
    }