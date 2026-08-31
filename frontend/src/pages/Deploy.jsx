import { useState, useEffect, useRef } from 'react'
import { IconRefresh, IconPlay, IconStop, IconRocket } from '../components/Icons'
import LongVideoDeploy from './LongVideoDeploy'

export default function Deploy({ targetId, target }) {
  const isVideo = target?.engine_type === 'comfyui'
  const [videoMode, setVideoMode] = useState('short') // short | long
  if (!isVideo) return <TextDeploy targetId={targetId} target={target} />
  return (
    <div>
      <div className="inline-flex gap-1 p-1 rounded-lg bg-card border border-gray/30 mb-6">
        {[['short', '短视频'], ['long', '长视频']].map(([k, label]) => (
          <button key={k} onClick={() => setVideoMode(k)}
            className={`px-4 py-1.5 rounded-md text-sm transition ${
              videoMode === k ? 'bg-blue text-bg font-semibold' : 'text-gray hover:text-fg'}`}>
            {label}
          </button>
        ))}
      </div>
      {videoMode === 'short'
        ? <VideoDeploy targetId={targetId} target={target} />
        : <LongVideoDeploy targetId={targetId} target={target} />}
    </div>
  )
}

/* ==================== 文本模型部署（llama.cpp / vLLM） ==================== */

function TextDeploy({ targetId }) {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState('')
  const [status, setStatus] = useState(null)
  const [msg, setMsg] = useState('')
  const [argsText, setArgsText] = useState('')
  const [argsMeta, setArgsMeta] = useState(null) // {source, score, ts, reasoning}
  const [loadingArgs, setLoadingArgs] = useState(false)

  useEffect(() => {
    if (!targetId) return
    setMsg('')
    fetch(`/api/deploy/models?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        setModels(d.models || [])
        setSelected(d.models?.[0] || '')
        if (d.error) setMsg(d.error)
      })
    fetch(`/api/deploy/status?target_id=${targetId}`)
      .then(r => r.json())
      .then(setStatus)
  }, [targetId])

  // 模型选定后拉取默认参数：优先最近调优，回退确定性生成
  useEffect(() => {
    if (!targetId || !selected) {
      setArgsText('')
      setArgsMeta(null)
      return
    }
    setLoadingArgs(true)
    fetch(`/api/deploy/default-args?target_id=${targetId}&model=${encodeURIComponent(selected)}`)
      .then(r => r.json())
      .then(d => {
        setArgsText(d.args || '')
        setArgsMeta(d)
      })
      .catch(() => setArgsMeta(null))
      .finally(() => setLoadingArgs(false))
  }, [targetId, selected])

  // Poll status until `running` flips to the expected value (or timeout).
  // Backend /start only sends the launch command without waiting for the engine
  // to be ready, so a single status query often returns false before the process
  // is detected, leaving the buttons in a stale state.
  async function pollStatus(expectRunning, timeoutMs = 90000, intervalMs = 1500) {
    const t0 = Date.now()
    while (true) {
      const s = await fetch(`/api/deploy/status?target_id=${targetId}`).then(r => r.json())
      setStatus(s)
      if (!!s.running === expectRunning) return
      if (Date.now() - t0 > timeoutMs) return
      await new Promise(r => setTimeout(r, intervalMs))
    }
  }

  async function start() {
    setMsg('启动中...')
    const res = await fetch('/api/deploy/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: targetId, model: selected, args_text: argsText }),
    })
    const d = await res.json()
    setMsg(d.message)
    if (d.success) await pollStatus(true)
  }

  async function stop() {
    setMsg('停止中...')
    const res = await fetch(`/api/deploy/stop?target_id=${targetId}`, { method: 'POST' })
    const d = await res.json()
    setMsg(d.message)
    if (d.success) await pollStatus(false)
  }

  function resetArgs() {
    if (!targetId || !selected) return
    setLoadingArgs(true)
    fetch(`/api/deploy/default-args?target_id=${targetId}&model=${encodeURIComponent(selected)}`)
      .then(r => r.json())
      .then(d => { setArgsText(d.args || ''); setArgsMeta(d) })
      .finally(() => setLoadingArgs(false))
  }

  const sourceLabel = {
    tuner: '自动调优回填',
    ai_tuner: 'AI 调优回填',
    generated: '硬件自适应生成',
    default: '引擎默认',
  }[argsMeta?.source] || ''

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">模型部署</h1>

      <div className="bg-card rounded-xl p-6 border border-gray/30 max-w-2xl">
        <div className="flex items-center gap-3 mb-6">
          <span className={`w-3 h-3 rounded-full ${status?.running ? 'bg-green' : 'bg-gray'}`} />
          <span className="font-semibold">{status?.running ? '运行中' : '未运行'}</span>
          {status?.engine && (
            <span className="text-xs px-2 py-0.5 rounded bg-blue/15 text-blue">
              引擎: {status.engine === 'vllm' ? 'vLLM' : 'llama.cpp'}
            </span>
          )}
        </div>

        <label className="block text-gray text-sm mb-2">选择模型</label>
        <select
          value={selected}
          onChange={e => setSelected(e.target.value)}
          className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 mb-2 text-fg"
        >
          {models.length === 0 && <option value="">（模型目录为空或未配置）</option>}
          {models.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <div className="text-xs text-gray/70 mb-6">共 {models.length} 个 .gguf 模型</div>

        <div className="flex items-center justify-between mb-2">
          <label className="block text-gray text-sm">运行参数</label>
          <button
            onClick={resetArgs}
            disabled={!selected || loadingArgs}
            className="text-xs text-blue hover:underline disabled:opacity-40"
          >
            {loadingArgs ? '加载参数中...' : (<span className="inline-flex items-center gap-1"><IconRefresh size={13} />恢复默认</span>)}
          </button>
        </div>
        {argsMeta && sourceLabel && (
          <div className="text-xs mb-2">
            <span className="text-gray">来源：</span>
            <span className={
              argsMeta.source === 'generated' || argsMeta.source === 'default'
                ? 'text-gray' : 'text-green'
            }>
              {sourceLabel}
            </span>
            {argsMeta.score > 0 && <span className="text-gray"> · 实测 {argsMeta.score} t/s</span>}
            {argsMeta.ts && <span className="text-gray/60"> · {argsMeta.ts}</span>}
          </div>
        )}
        <textarea
          value={argsText}
          onChange={e => setArgsText(e.target.value)}
          spellCheck={false}
          placeholder="--ctx-size 8192 --n-gpu-layers all --batch-size 4096 ..."
          className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-fg font-mono text-xs h-24 focus:border-blue outline-none resize-y"
        />
        <div className="text-xs text-gray/60 mb-6">
          可手动编辑。端口 / 监听地址 / 监控开关由系统自动注入，无需在此填写。
        </div>

        <div className="flex gap-3">
          <button
            onClick={start}
            disabled={status?.running || !selected}
            className="flex-1 bg-green text-bg font-bold py-2 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition"
          >
            <span className="inline-flex items-center justify-center gap-1.5"><IconPlay size={14} />启动</span>
          </button>
          <button
            onClick={stop}
            disabled={!status?.running}
            className="flex-1 bg-red text-bg font-bold py-2 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition"
          >
            <span className="inline-flex items-center justify-center gap-1.5"><IconStop size={13} />停止</span>
          </button>
        </div>

        {msg && <div className="mt-4 text-sm text-gray">{msg}</div>}
      </div>
    </div>
  )
}

/* ==================== 视频模型部署（ComfyUI） ==================== */

const RES_PRESETS = [
  { label: '480p (832×480)', width: 832, height: 480 },
  { label: '720p (1280×720)', width: 1280, height: 720 },
  { label: '竖屏 (480×832)', width: 480, height: 832 },
]

function VideoDeploy({ targetId, target }) {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState('')
  const [status, setStatus] = useState(null)
  const [msg, setMsg] = useState('')
  const [form, setForm] = useState({
    prompt: '',
    negative_prompt: '',
    width: 832,
    height: 480,
    length: 49,
    steps: 30,
    cfg: 6.0,
    seed: '',
    fps: 16,
    enhance: true,
  })
  const [job, setJob] = useState(null) // {prompt_id, state, files}
  const pollRef = useRef(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  useEffect(() => {
    if (!targetId) return
    setMsg('')
    fetch(`/api/deploy/status?target_id=${targetId}`).then(r => r.json()).then(setStatus)
    // 扫描目标机 ComfyUI diffusion_models 目录，列出真实已下载的模型
    fetch(`/api/deploy/video-models?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        const list = d.models || []
        setModels(list)
        setSelected(list[0] ? list[0].filename : '')
      })
  }, [targetId])

  // 轮询生成进度
  useEffect(() => {
    if (!job?.prompt_id || job.state === 'completed') return
    pollRef.current = setInterval(async () => {
      const d = await fetch(
        `/api/deploy/generate/progress?target_id=${targetId}&prompt_id=${job.prompt_id}`
      ).then(r => r.json())
      setJob(j => (j ? { ...j, ...d } : j))
      if (d.state === 'completed') clearInterval(pollRef.current)
    }, 3000)
    return () => clearInterval(pollRef.current)
  }, [job?.prompt_id])

  // Poll status until `running` flips to the expected value (or timeout).
  async function pollStatus(expectRunning, timeoutMs = 90000, intervalMs = 1500) {
    const t0 = Date.now()
    while (true) {
      const s = await fetch(`/api/deploy/status?target_id=${targetId}`).then(r => r.json())
      setStatus(s)
      if (!!s.running === expectRunning) return
      if (Date.now() - t0 > timeoutMs) return
      await new Promise(r => setTimeout(r, intervalMs))
    }
  }

  async function startEngine() {
    setMsg('启动 ComfyUI 中...')
    const res = await fetch('/api/deploy/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: targetId, model: selected, args_text: '' }),
    })
    const d = await res.json()
    setMsg(d.message)
    if (d.success) await pollStatus(true)
  }

  async function stopEngine() {
    setMsg('停止中...')
    const res = await fetch(`/api/deploy/stop?target_id=${targetId}`, { method: 'POST' })
    const d = await res.json()
    setMsg(d.message)
    if (d.success) await pollStatus(false)
  }

  async function generate() {
    setMsg('')
    const res = await fetch('/api/deploy/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_id: targetId,
        prompt: form.prompt,
        model_name: selected,
        negative_prompt: form.negative_prompt,
        width: Number(form.width),
        height: Number(form.height),
        length: Number(form.length),
        steps: Number(form.steps),
        cfg: Number(form.cfg),
        seed: form.seed === '' ? null : Number(form.seed),
        fps: Number(form.fps),
        enhance: !!form.enhance,
      }),
    })
    const d = await res.json()
    if (!d.success) { setMsg(d.message || '提交失败'); return }
    setJob({
      prompt_id: d.prompt_id,
      state: 'queued',
      enhanced: d.enhanced,
      final_prompt: d.final_prompt,
      reasoning: d.reasoning,
    })
  }

  const stateLabel = {
    queued: '排队中…',
    running: '生成中…（首次加载模型较慢，请耐心等待）',
    completed: '生成完成',
    unknown: '状态未知',
    error: '出错',
  }[job?.state] || ''

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">视频生成</h1>

      <div className="bg-card rounded-xl p-6 border border-gray/30 max-w-2xl">
        <div className="flex items-center gap-3 mb-6">
          <span className={`w-3 h-3 rounded-full ${status?.running ? 'bg-green' : 'bg-gray'}`} />
          <span className="font-semibold">{status?.running ? 'ComfyUI 运行中' : 'ComfyUI 未运行'}</span>
          <span className="text-xs px-2 py-0.5 rounded bg-purple/15 text-purple">引擎: ComfyUI</span>
        </div>

        {!status?.running ? (
          <div className="mb-6">
            <div className="text-sm text-gray mb-3">
              视频生成需先启动 ComfyUI 服务。请确认已在设置中配置 ComfyUI 安装目录，如未安装可一键安装。
            </div>
            <div className="flex gap-3">
              <button
                onClick={startEngine}
                className="flex-1 bg-green text-bg font-bold py-2 rounded-lg hover:opacity-90 transition"
              >
                <span className="inline-flex items-center justify-center gap-1.5"><IconPlay size={14} />启动 ComfyUI</span>
              </button>
              <button
                onClick={stopEngine}
                disabled={!status?.running}
                className="flex-1 bg-red text-bg font-bold py-2 rounded-lg disabled:opacity-40 hover:opacity-90 transition"
              >
                <span className="inline-flex items-center justify-center gap-1.5"><IconStop size={13} />停止</span>
              </button>
            </div>
            {msg && <div className="mt-3 text-sm text-gray">{msg}</div>}
          </div>
        ) : (
          <>
            <label className="block text-gray text-sm mb-2">视频模型</label>
            <select
              value={selected}
              onChange={e => setSelected(e.target.value)}
              className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 mb-1 text-fg"
            >
              {models.length === 0 && <option value="">（无可用视频模型，请先到商店下载）</option>}
              {models.map(m => (
                <option key={m.filename} value={m.filename}>
                  {m.name}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray/70 mb-5">共 {models.length} 个视频模型</div>

            <label className="block text-gray text-sm mb-2">画面描述（Prompt）</label>
            <textarea
              value={form.prompt}
              onChange={e => set('prompt', e.target.value)}
              placeholder="一只在雪地里奔跑的柴犬，电影质感，慢动作"
              className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-fg text-sm h-20 focus:border-blue outline-none resize-y mb-2"
            />

            {/* AI 智能编排开关 */}
            <button
              type="button"
              onClick={() => set('enhance', !form.enhance)}
              className="flex items-center gap-2 mb-5 group"
            >
              <span className={`relative w-9 h-5 rounded-full transition-colors ${
                form.enhance ? 'bg-blue' : 'bg-gray/40'
              }`}>
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-bg transition-transform ${
                  form.enhance ? 'translate-x-4' : ''
                }`} />
              </span>
              <span className="text-sm text-fg">AI 智能编排</span>
              <span className="text-xs text-gray/70">
                {form.enhance ? '大模型将把你的描述扩写成电影级提示词并推荐参数' : '直接使用你填写的提示词与参数'}
              </span>
            </button>

            <label className="block text-gray text-sm mb-2">分辨率</label>
            <div className="flex gap-2 mb-4">
              {RES_PRESETS.map(r => (
                <button
                  key={r.label}
                  onClick={() => { set('width', r.width); set('height', r.height) }}
                  className={`px-3 py-1.5 rounded-lg text-sm border transition ${
                    form.width === r.width && form.height === r.height
                      ? 'border-blue bg-blue/15 text-blue'
                      : 'border-gray/40 text-gray hover:border-gray'
                  }`}
                >
                  {r.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <Field label="帧数" value={form.length} onChange={v => set('length', v)} hint="≈时长×fps" />
              <Field label="步数" value={form.steps} onChange={v => set('steps', v)} hint="越高越清晰" />
              <Field label="CFG" value={form.cfg} onChange={v => set('cfg', v)} hint="提示词强度" />
            </div>
            <div className="grid grid-cols-2 gap-3 mb-6">
              <Field label="帧率 fps" value={form.fps} onChange={v => set('fps', v)} />
              <Field label="种子" value={form.seed} onChange={v => set('seed', v)} placeholder="留空随机" />
            </div>

            <button
              onClick={generate}
              disabled={!form.prompt || !selected}
              className="w-full bg-gradient-to-r from-blue to-purple text-bg font-bold py-2.5 rounded-lg disabled:opacity-40 hover:opacity-90 transition"
            >
              <span className="inline-flex items-center justify-center gap-1.5"><IconRocket size={15} />生成视频</span>
            </button>
            {msg && <div className="mt-3 text-sm text-red">{msg}</div>}
          </>
        )}

        {/* 生成进度 */}
        {job && (
          <div className="mt-6 p-4 rounded-lg bg-bg border border-gray/30">
            <div className="flex items-center gap-2 mb-2">
              {job.state !== 'completed' && (
                <span className="w-2 h-2 rounded-full bg-blue animate-pulse" />
              )}
              <span className="text-sm font-semibold text-fg">{stateLabel}</span>
            </div>
            {job.enhanced && (
              <div className="mb-3 p-3 rounded-lg bg-blue/5 border border-blue/20">
                <div className="text-xs font-semibold text-blue mb-1">AI 已编排提示词</div>
                {job.final_prompt && (
                  <div className="text-xs text-fg/80 italic leading-relaxed mb-1">“{job.final_prompt}”</div>
                )}
                {job.reasoning && (
                  <div className="text-[11px] text-gray/70 leading-relaxed">{job.reasoning}</div>
                )}
              </div>
            )}
            {job.state === 'completed' && job.files?.length > 0 && (
              <div className="text-xs text-gray">
                {job.files.map((f, i) => {
                  const src = `/api/deploy/video-file?target_id=${targetId}`
                    + `&filename=${encodeURIComponent(f.filename)}`
                    + `&subfolder=${encodeURIComponent(f.subfolder || '')}`
                  return (
                    <div key={i} className="mb-3">
                      <video
                        src={src}
                        controls
                        autoPlay
                        loop
                        playsInline
                        className="w-full rounded-lg border border-gray/30 bg-black"
                        style={{ maxHeight: '420px' }}
                      />
                      <div className="mt-1 font-mono break-all text-gray/60">
                        {f.subfolder ? `${f.subfolder}/` : ''}{f.filename}
                      </div>
                    </div>
                  )
                })}
                <div className="mt-1 text-gray/60">文件保存在目标机 ComfyUI/output 目录</div>
              </div>
            )}
            {job.state === 'completed' && (!job.files || job.files.length === 0) && (
              <div className="text-xs text-gray">任务已完成，但未检测到输出文件（请检查 ComfyUI 日志）</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Field({ label, value, onChange, hint, placeholder }) {
  return (
    <div>
      <label className="block text-gray text-xs mb-1">{label}</label>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-bg border border-gray/40 rounded-lg px-2 py-1.5 text-fg text-sm focus:border-blue outline-none"
      />
      {hint && <div className="text-[10px] text-gray/50 mt-0.5">{hint}</div>}
    </div>
  )
}
