"""ComfyUI 引擎适配器（视频 / 图像生成）

ComfyUI 与 llama.cpp / vLLM 的交互范式根本不同：它不是"加载模型后持续推理
吐 token"，而是"HTTP 服务常驻 → 提交一个 workflow 节点图 JSON → 排队异步生成
→ 轮询 history 取产物文件"。因此本适配器除实现 EngineAdapter 基础契约外，
额外提供提交生成任务、查询进度 / 产物 / 显存的方法。

关键约束：
  - ComfyUI 监听在目标机本地（默认 127.0.0.1:<port>），本机（控制端）访问不到，
    所有 HTTP 请求必须在目标机上用 curl 发起（与 collectors 采集 metrics 同套路）。
  - workflow JSON 体积可能较大，统一用 write_file 落盘 + curl @file，绕开命令行
    长度限制（Windows cmd 8191 字符上限，见已验证的 _curl_completion 经验）。
  - 不硬编码任何个人环境：安装目录 / python 入口 / 端口全部来自 Target 配置。

所有命令基于用户配置的 Target 执行。
"""

import base64
import json
import shlex
import time
import uuid
from typing import Optional

from .engine_adapter import EngineAdapter, StartParams
from .executor import Executor
from ..models.target import Target


def _path_join(target: Target, base: str, name: str) -> str:
    """按目标 OS 拼接路径（base 为目录，name 为文件名）。"""
    sep = "\\" if target.os == "windows" else "/"
    return base.rstrip("\\/") + sep + name

# ComfyUI 默认服务端口（Target.service_port 未显式配置时回退）
DEFAULT_COMFY_PORT = 8188


def _comfy_port(target: Target) -> int:
    return target.service_port or DEFAULT_COMFY_PORT


class ComfyUIAdapter(EngineAdapter):
    def __init__(self, executor: Executor, target: Target):
        self.executor = executor
        self.target = target

    def name(self) -> str:
        return "comfyui"

    # ==================== 路径 / 命令解析 ====================

    def _comfy_dir(self) -> str:
        """ComfyUI 安装根目录：来自 engine_path（用户配置），否则回退常见默认。
        注意：这里不写死任何个人机器路径，回退值只是通用约定位置。"""
        return self.target.engine_path or ""

    def _python_cmd(self) -> str:
        """启动用的 python 解释器。注意：engine_path 是 ComfyUI 安装根目录，
        绝不能当 python 解释器用（历史 bug 导致启动命令把目录名当可执行文件，
        进程起不来）。这里回退到 PATH 中的 python；Windows 便携版由 _start_windows
        的 bat 内联探测 python_embeded/python 子目录覆盖。"""
        return "python"

    def _base_url(self) -> str:
        return f"http://127.0.0.1:{_comfy_port(self.target)}"

    # ==================== 检测 ====================

    def check_installed(self) -> bool:
        """检测 ComfyUI 是否已安装：main.py 入口文件是否存在。"""
        d = self._comfy_dir()
        if not d:
            return False
        main_py = _path_join(self.target, d, "main.py")
        if self.target.os == "windows":
            result = self.executor.run(f'if exist "{main_py}" (echo FOUND)')
        else:
            result = self.executor.run(f'test -f "{main_py}" && echo FOUND')
        return "FOUND" in result.stdout

    # ==================== 启动 / 停止 ====================

    def start(self, params: StartParams) -> tuple:
        """后台启动 ComfyUI 服务。params.model_path 在 ComfyUI 语义下不使用
        （模型由 workflow 指定），这里仅用于日志展示。"""
        port = _comfy_port(self.target)
        d = self._comfy_dir()
        if not d:
            return False, "未配置 ComfyUI 安装目录（engine_path 应指向 ComfyUI 根目录）"
        main_py = _path_join(self.target, d, "main.py")
        # --listen 0.0.0.0 便于本机端口转发访问；--port 指定端口
        # --disable-async-offload --disable-mmap：绕开 comfy_aimdo 异步权重 I/O
        # 后端在 VAE 解码读 safetensors 分片时的 read_file_slice failed（实测
        # RTX4090+H3 必现，关掉后回退普通加载，采样+解码全通过并成功出片）。
        # --disable-smart-memory：禁用智能显存管理，模型加载后不主动换出。
        # 长视频多段场景关键：smart-memory 会在任务间把 video_vae 换出显存，
        # 下一段 I2V 冷加载 vae.encode 时又撞 aimdo read_file_slice failed；
        # 常驻不换出即从根上不触发该冷加载路径。
        run_args = (
            f'"{main_py}" --listen 0.0.0.0 --port {port} '
            '--disable-async-offload --disable-mmap --disable-smart-memory'
        )

        if self.target.os == "windows":
            return self._start_windows(d, run_args)
        return self._start_linux(d, run_args)

    def _start_windows(self, d: str, run_args: str) -> tuple:
        # bat 内联探测 python：优先便携版 python_embeded/python.exe，
        # 其次 python/python.exe，最后回退 PATH 中的 python。避免把 ComfyUI
        # 目录名当解释器（历史 bug）或 PATH 无 python 导致进程起不来。
        bat_content = (
            '@echo off\r\n'
            f'cd /d "{d}"\r\n'
            'set "PY=python"\r\n'
            f'if exist "{d}\\python_embeded\\python.exe" set "PY={d}\\python_embeded\\python.exe"\r\n'
            f'if exist "{d}\\python\\python.exe" set "PY={d}\\python\\python.exe"\r\n'
            f'"%PY%" {run_args}\r\n'
        )
        b64 = base64.b64encode(bat_content.encode("gbk")).decode("ascii")
        bat_path = r"C:\temp\comfy_start.bat"
        write_cmd = (
            'powershell -Command "'
            "New-Item -Path C:\\temp -ItemType Directory -Force | Out-Null; "
            f"[IO.File]::WriteAllBytes('{bat_path}', [Convert]::FromBase64String('{b64}'))"
            '"'
        )
        result = self.executor.run(write_cmd, timeout=15)
        if not result.ok:
            return False, f"写入启动脚本失败: {result.stdout} {result.stderr}"
        run_cmd = (
            'schtasks /create /tn ComfyUI /tr "%s" /sc once /st 00:00 /f '
            '&& schtasks /run /tn ComfyUI' % bat_path
        )
        result = self.executor.run(run_cmd, timeout=15)
        if not result.ok:
            return False, f"启动失败: {result.stdout} {result.stderr}"
        return True, "ComfyUI 启动命令已发送（首次启动需加载依赖，请耐心等待）"

    def _start_linux(self, d: str, run_args: str) -> tuple:
        py = self._python_cmd()
        cmd = f'cd "{d}" && nohup {py} {run_args} > /tmp/comfyui.log 2>&1 &'
        result = self.executor.run(cmd, timeout=20)
        if not result.ok:
            return False, f"启动失败: {result.stdout} {result.stderr}"
        return True, "ComfyUI 启动命令已发送"

    def stop(self) -> tuple:
        if self.target.os == "windows":
            self.executor.run('schtasks /end /tn ComfyUI', timeout=10)
            result = self.executor.run("taskkill /f /im python.exe", timeout=10)
            # taskkill python.exe 过宽，但 ComfyUI 在 Windows 通常就是 python 进程；
            # 更精确需按端口杀，这里保持与 llama 一致的尽力而为策略
        else:
            result = self.executor.run("pkill -f 'ComfyUI/main.py'", timeout=10)
        if result.ok:
            return True, "ComfyUI 服务已停止"
        return False, f"停止结果: {result.stdout} {result.stderr}"

    def is_running(self) -> bool:
        # 健康检查：ComfyUI 提供 /system_stats，能连通即视为运行中
        return self._curl_json("/system_stats", timeout=8) is not None

    def get_metrics_url(self) -> str:
        return f"{self._base_url()}/system_stats"

    # ==================== 生成任务（ComfyUI 专属） ====================

    def _remote_tmp(self, name: str) -> str:
        if self.target.os == "windows":
            return f"C:\\temp\\{name}"
        return f"/tmp/{name}"

    def _curl_json(self, path: str, timeout: int = 10) -> Optional[dict]:
        """在目标机上 curl 一个 GET 端点并解析 JSON；失败返回 None。"""
        url = f"{self._base_url()}{path}"
        if self.target.os == "windows":
            cmd = f'curl -s --max-time {timeout} "{url}"'
        else:
            cmd = f"curl -s --max-time {timeout} {shlex.quote(url)}"
        result = self.executor.run(cmd, timeout=timeout + 5)
        out = (result.stdout or "").strip()
        if not out or not out.startswith("{"):
            return None
        try:
            return json.loads(out)
        except (ValueError, json.JSONDecodeError):
            return None

    def submit_workflow(self, workflow: dict, client_id: Optional[str] = None) -> tuple:
        """提交一个 workflow 节点图 JSON 到 /prompt，返回 (ok, prompt_id 或错误)。

        workflow 已是 ComfyUI 标准 prompt 格式（非 UI 导出的 litegraph 格式）。
        大 JSON 用 write_file 落盘 + curl @file，避免命令行截断。"""
        cid = client_id or uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": cid}
        body = json.dumps(payload, ensure_ascii=False)
        tmp = self._remote_tmp(f"comfy_prompt_{cid}.json")
        if not self.executor.write_file(body, tmp):
            return False, "写入 workflow 临时文件失败"

        url = f"{self._base_url()}/prompt"
        if self.target.os == "windows":
            cmd = f'curl -s --max-time 30 -X POST "{url}" -H "Content-Type: application/json" --data "@{tmp}"'
        else:
            cmd = f"curl -s --max-time 30 -X POST {shlex.quote(url)} -H 'Content-Type: application/json' --data @{shlex.quote(tmp)}"
        result = self.executor.run(cmd, timeout=35)
        out = (result.stdout or "").strip()
        try:
            data = json.loads(out)
        except (ValueError, json.JSONDecodeError):
            return False, f"提交失败，响应无法解析: {out[:300]}"
        if "prompt_id" in data:
            return True, data["prompt_id"]
        # ComfyUI 校验失败会返回 error 详情
        err = data.get("error") or data
        return False, f"workflow 被拒绝: {json.dumps(err, ensure_ascii=False)[:400]}"

    def get_history(self, prompt_id: str) -> Optional[dict]:
        """查询任务历史；任务完成后 outputs 里含产物（视频/图片）文件信息。"""
        return self._curl_json(f"/history/{prompt_id}", timeout=10)

    def get_queue(self) -> Optional[dict]:
        """查询队列：running + pending，用于判断任务是否在跑。"""
        return self._curl_json("/queue", timeout=8)

    def get_progress(self, prompt_id: str) -> dict:
        """返回任务粗粒度状态（不依赖 WebSocket，纯轮询）：
        {state: queued|running|completed|unknown, ...}

        ComfyUI 的逐步进度(step/total)只在 WebSocket 推送，HTTP 侧无法直接拿；
        P0 用队列 + history 推断状态，进度条以'排队/生成中/完成'三态呈现。"""
        hist = self.get_history(prompt_id)
        if hist and prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {}) or {}
            # 关键：history 里有记录 ≠ 成功。ComfyUI 执行报错时也会写入 history，
            # 必须先看 status_str，否则失败段会被误判成 completed（outputs 为空）。
            if status.get("status_str") == "error":
                msg = ""
                for m in status.get("messages", []) or []:
                    if isinstance(m, list) and len(m) > 1 and m[0] == "execution_error":
                        info = m[1] or {}
                        msg = "%s @node %s" % (
                            info.get("exception_message", "").strip(),
                            info.get("node_type", ""))
                        break
                return {"state": "error", "message": msg or "ComfyUI 执行报错"}
            outputs = entry.get("outputs", {})
            return {"state": "completed", "outputs": outputs}
        q = self.get_queue()
        if q:
            for item in q.get("queue_running", []):
                # item[1] 是 prompt_id
                if len(item) > 1 and item[1] == prompt_id:
                    return {"state": "running"}
            for item in q.get("queue_pending", []):
                if len(item) > 1 and item[1] == prompt_id:
                    return {"state": "queued"}
        return {"state": "unknown"}

    def get_system_stats(self) -> Optional[dict]:
        """显存 / 设备信息：/system_stats 返回 devices 列表含 vram_total/vram_free。"""
        return self._curl_json("/system_stats", timeout=8)

    # ==================== 健康等待辅助 ====================

    def wait_ready(self, max_wait: int = 60) -> bool:
        """轮询直到 ComfyUI 可响应（首次启动加载依赖较慢时用）。"""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self.is_running():
                return True
            time.sleep(2)
        return False


    # ==================== 视频生成 workflow 模板（MiniMax H3） ====================

    # H3 默认权重文件名（Comfy-Org/MiniMax-H3 重打包，24G 显存极限压缩组合）
    H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    H3_VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
    H3_VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
    H3_LORA_8STEP = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

    def build_video_workflow(
        self,
        prompt: str,
        model_name: str = "",
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        length: int = 49,
        steps: int = 8,
        cfg: float = 1.0,
        seed: Optional[int] = None,
        fps: int = 24,
        image_name: str = "",
        ref_image_names: Optional[list] = None,
        teacache: bool = False,
        teacache_thresh: float = 0.15,
    ) -> dict:
        """构建 MiniMax H3 text-to-video / image-to-video 的 ComfyUI API prompt（扁平格式）。

        image_name 非空时走 I2V：在 ComfyUI input 目录里加载该首帧图，连到
        MiniMaxH3ImageToVideo 的 first_frame 端口（该节点内部自 vae.encode，
        无需额外 VAEEncode 节点）。为空则是纯 T2V。

        官方 T2V workflow 用 ComfyUI 新版「子图(subgraph)」封装生成逻辑，
        /prompt 端点只接受扁平 API 格式，故这里把子图内部 21 节点展平为
        顶层 API prompt，并把子图对外参数（prompt/分辨率/时长/seed/权重名）
        注入到对应节点。结构依据 Comfy-Org/workflow_templates 的
        video_minimax_h3_t2v.json 实测解析（节点 119-139 + SaveVideo）。

        参数：
          prompt    正向提示词（H3 无独立负面提示词节点，negative 忽略）
          duration  时长秒（H3 用公式换算帧数 length，非直接帧数）
          steps     采样步数（turbo lora 推荐 8）
          seed      随机种子（None 则随机）
        权重文件名固定为 24G 显存极限压缩组合（pruned+int8_convrot UNet、
        nvfp4 文本编码器、双 VAE、8step turbo lora），如需换档改类常量。
        """
        sd = seed if seed is not None else int(uuid.uuid4().int % (2 ** 32 - 1))
        unet = model_name or self.H3_UNET
        wf = {
            "119": {"class_type": "VAELoader",
                    "inputs": {"vae_name": self.H3_VAE_VIDEO}},
            "120": {"class_type": "VAELoader",
                    "inputs": {"vae_name": self.H3_VAE_AUDIO}},
            "127": {"class_type": "UNETLoader",
                    "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "128": {"class_type": "CLIPLoader",
                    "inputs": {"clip_name": self.H3_CLIP, "type": "minimax", "device": "default"}},
            "134": {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["127", 0], "lora_name": self.H3_LORA_8STEP,
                               "strength_model": 1.0}},
            # use_turbo=False -> switch 走 on_false(原始 UNet)；True -> on_true(LoRA)
            "139": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
            "135": {"class_type": "ComfySwitchNode",
                    "inputs": {"on_false": ["127", 0], "on_true": ["134", 0], "switch": ["139", 0]}},
            # steps 切换：turbo 用 138(注入 steps)，非 turbo 用 137(固定 20)
            "138": {"class_type": "PrimitiveInt", "inputs": {"value": steps}},
            "137": {"class_type": "PrimitiveInt", "inputs": {"value": 20}},
            "136": {"class_type": "ComfySwitchNode",
                    "inputs": {"on_false": ["137", 0], "on_true": ["138", 0], "switch": ["139", 0]}},
            "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
            "124": {"class_type": "BasicScheduler",
                    "inputs": {"model": ["135", 0], "scheduler": "simple",
                               "steps": ["136", 0], "denoise": 1.0}},
            "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": sd}},
            # 帧数 length -> H3 合法的 17 对齐帧数（PrimitiveFloat=length -> MathExpression）
            "133": {"class_type": "PrimitiveFloat", "inputs": {"value": float(length)}},
            "132": {"class_type": "ComfyMathExpression",
                    "inputs": {"expression": "max(5, a) + (5 - (max(5, a) % 17)) % 17",
                               "values.a": ["133", 0]}},
            "131": {"class_type": "MiniMaxH3ImageToVideo",
                    "inputs": {"clip": ["128", 0], "vae": ["119", 0], "prompt": prompt,
                               "width": width, "height": height, "length": ["132", 1]}},
            "126": {"class_type": "BasicGuider",
                    "inputs": {"model": ["135", 0], "conditioning": ["131", 0]}},
            "125": {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["129", 0], "guider": ["126", 0], "sampler": ["123", 0],
                               "sigmas": ["124", 0], "latent_image": ["131", 1]}},
            "122": {"class_type": "VAEDecode",
                    "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
            "121": {"class_type": "VAEDecodeAudio",
                    "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
            "130": {"class_type": "CreateVideo",
                    "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": fps}},
            "92": {"class_type": "SaveVideo",
                   "inputs": {"video": ["130", 0], "filename_prefix": "modeldeploy/video",
                              "format": "auto"}},
        }
        # R2V：有参考图时，把节点 131 换成 MiniMaxH3ReferenceToVideo，用
        # ref_image_N autogrow 端口连多张 LoadImage。身份由模型内部对齐
        # （参考 token 随每个采样步贯穿），比"逐镜预生成锚帧"可靠——文生图
        # 跨图人脸不一致，而 R2V 直接拿参考图锁身份。prompt 用 <Picture i> 引用。
        # 注意：R2V 比 I2V 多一个 audio_vae 输入（节点 120 工作流里已存在）。
        if ref_image_names:
            wf["131"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                         "inputs": {"clip": ["128", 0], "vae": ["119", 0],
                                    "audio_vae": ["120", 0], "prompt": prompt,
                                    "width": width, "height": height,
                                    "length": ["132", 1],
                                    "ref_image_size": "match"}}
            # ref_images 是 Autogrow 输入，API 格式须序列化为嵌套 dict
            # {ref_image_0: [node,port], ref_image_1: ...}，不能把 ref_image_0
            # 当顶层参数（否则 execute 收到 unexpected kwarg 'ref_image_0'）。
            ref_map = {}
            for i, name in enumerate(ref_image_names[:9]):
                nid = "15%d" % i  # 150,151,...
                wf[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
                ref_map["ref_image_%d" % i] = [nid, 0]
            wf["131"]["inputs"]["ref_images"] = ref_map
        elif image_name:
            # I2V：有首帧图时挂 LoadImage 节点，连到 MiniMaxH3ImageToVideo.first_frame
            wf["140"] = {"class_type": "LoadImage",
                         "inputs": {"image": image_name}}
            wf["131"]["inputs"]["first_frame"] = ["140", 0]
        # TeaCache：在最终模型(135)与 guider/scheduler 之间插缓存节点，跳过相邻
        # 冗余去噪步。total_steps 必须等于实际采样步数(steps)，否则缓存窗口错位。
        # start_step=2/end_step=-2：首 2 步(定结构)与末 2 步(定细节)始终真实计算。
        if teacache:
            wf["145"] = {"class_type": "MiniMaxH3TeaCache",
                         "inputs": {"model": ["135", 0],
                                    "rel_l1_thresh": teacache_thresh,
                                    "start_step": 2, "end_step": -2,
                                    "total_steps": int(steps)}}
            wf["126"]["inputs"]["model"] = ["145", 0]
            wf["124"]["inputs"]["model"] = ["145", 0]
        return wf

    def build_upscale_workflow(
        self,
        image_name: str,
        out_w: int = 1920,
        out_h: int = 1080,
        model_name: str = "RealESRGAN_x4plus.pth",
        filename_prefix: str = "modeldeploy/upscaled",
    ) -> dict:
        """构建单图超分 ComfyUI API prompt（扁平格式）。

        链路：LoadImage → UpscaleModelLoader → ImageUpscaleWithModel
        → ImageScale(精确缩到 out_w×out_h) → SaveImage。

        RealESRGAN_x4plus 固定 4× 放大，832×480 会先变 3328×1920，再用
        ImageScale(lancos/downscale) 收敛到目标 1920×1080，避免尺寸失控。
        image_name 须是 ComfyUI input 目录里已存在的图（参考图/抽帧图）。
        """
        return {
            "200": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "201": {"class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": model_name}},
            "202": {"class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["201", 0], "image": ["200", 0]}},
            "203": {"class_type": "ImageScale",
                    "inputs": {"image": ["202", 0], "upscale_method": "lanczos",
                               "downscale_method": "lanczos",
                               "width": int(out_w), "height": int(out_h),
                               "crop": "disabled"}},
            "204": {"class_type": "SaveImage",
                    "inputs": {"images": ["203", 0], "filename_prefix": filename_prefix}},
        }
