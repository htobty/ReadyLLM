import { useState, useEffect, useRef } from 'react'
import ProgressPanel from '../components/ProgressPanel'
import { IconAlert, IconStar, IconCheck, IconX, IconBot } from '../components/Icons'

// ==================== 自动调优 ====================

const GOALS = [
  ['latency', '端到端体感', '综合首字延迟+生成速度，适合对话'],
  ['throughput', '解码吞吐', '纯追求生成 t/s 最高'],
  ['prefill', '长文本预填充', '长输入处理最快'],
]

const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

function fmtCtx(c) {
  if (c >= 1048576) return `${(c / 1048576).toFixed(c % 1048576 ? 1 : 0)}M`
  if (c >= 1024) return `${Math.round(c / 1024)}K`
  return String(c)
}

// 上下文长度选择器：预设快捷值 + 可自定义输入任意 token 数（不固定）
function CtxPicker({ value, onChange }) {
  return (
    <div className="mb-4">
      <div className="flex flex-wrap gap-2 mb-2">
        {CTX_PRESETS.map(c => (
          <button key={c} type="button" onClick={() => onChange(c)}
            className={`px-3 py-1 rounded-lg text-sm border transition ${Number(value) === c ? 'border-blue bg-blue/20 text-blue' : 'border-gray/40 text-gray hover:bg-gray/10'}`}>
            {fmtCtx(c)}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray whitespace-nowrap">自定义</span>
        <input type="number" min={1024} step={1024} value={value}
          onChange={e => onChange(Number(e.target.value) || 0)}
          className="flex-1 bg-bg border border-gray/40 rounded-lg px-3 py-1.5 text-fg text-sm" />
        <span className="text-xs text-gray">tokens</span>
      </div>
    </div>
  )
}

const DEFAULT_BASELINE = {
  'spec-type': 'draft-mtp',
  'cache-type-k': 'q4_0',
  'cache-type-v': 'q4_0',
  'n-gpu-layers': 'all',
  'batch-size': '4096',
  'ubatch-size': '1024',
  'spec-draft-n-max': '3',
}

function AutoTune({ targetId }) {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState('')
  const [goal, setGoal] = useState('latency')
  const [ctxSize, setCtxSize] = useState(8192)
  const [useBaseline, setUseBaseline] = useState(true)
  const [state, setState] = useState('idle')
  const [logs, setLogs] = useState([])
  const [results, setResults] = useState([])
  const [best, setBest] = useState(null)
  const [baseline, setBaseline] = useState(null)
  const [error, setError] = useState('')
  const pollRef = useRef(null)

  function resumePoll(jobId) {
    clearInterval(pollRef.current)
    setState('running')
    pollRef.current = setInterval(async () => {
      const pr = await fetch(`/api/tune/status/${jobId}`)
      const job = await pr.json()
      setLogs(job.logs || [])
      if (job.status === 'success' || job.status === 'failed') {
        clearInterval(pollRef.current)
        setResults(job.results || [])
        setBest(job.best || null)
        setBaseline(job.baseline || null)
        setState(job.status)
        if (job.error) setError(job.error)
      }
    }, 2500)
  }

  useEffect(() => {
    if (!targetId) return
    setState('idle'); setResults([]); setLogs([]); setError('')
    setBest(null); setBaseline(null)
    fetch(`/api/store/downloaded?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        const list = (d.models || []).map(m => m.filename)
        setModels(list)
        setSelected(list[0] || '')
      })
    // 刷新/重进页面后恢复正在运行的调优任务
    fetch(`/api/tune/active?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        const jobs = d.jobs || []
        if (jobs.length > 0) {
          const j = jobs[0]
          if (j.model) setSelected(j.model)
          if (j.ctx_size) setCtxSize(j.ctx_size)
          if (j.goal) setGoal(j.goal)
          setLogs(j.last_logs || [])
          resumePoll(j.job_id)
        }
      })
      .catch(() => {})
    return () => clearInterval(pollRef.current)
  }, [targetId])

  async function start() {
    setState('running')
    setLogs([]); setResults([]); setError(''); setBest(null); setBaseline(null)
    const res = await fetch('/api/tune/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_id: targetId,
        model: selected,
        ctx_size: ctxSize,
        goal,
        baseline_cfg: useBaseline ? DEFAULT_BASELINE : null,
      }),
    })
    const d = await res.json()
    if (!d.ok) { setState('failed'); setError(d.message || '启动失败'); return }
    resumePoll(d.job_id)
  }

  let gain = null
  if (best && baseline && baseline.metrics?.decode > 0 && best.metrics?.decode > 0) {
    gain = ((best.metrics.decode - baseline.metrics.decode) / baseline.metrics.decode * 100).toFixed(1)
  }

  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-6">
      <div className="bg-card rounded-xl p-6 border border-gray/30 lg:col-span-3">
        <p className="text-gray text-sm mb-4">
          在你当前硬件下两阶段搜索最优参数：先定主导因素（投机/缓存/卸载），再收敛微调（batch/ubatch/threads）。
          每组多次测速取中位数。完整调优约需 15-30 分钟。
        </p>

        <label className="block text-gray text-sm mb-2">选择模型</label>
        <select value={selected} onChange={e => setSelected(e.target.value)}
          className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 mb-4 text-fg">
          {models.length === 0 && <option value="">（模型目录为空）</option>}
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>

        <label className="block text-gray text-sm mb-2">上下文长度（固定约束，AI 调优不会改动它）</label>
        <CtxPicker value={ctxSize} onChange={setCtxSize} />

        <label className="block text-gray text-sm mb-2">优化目标</label>
        <div className="grid grid-cols-3 gap-3 mb-4">
          {GOALS.map(([v, l, hint]) => (
            <button key={v} onClick={() => setGoal(v)}
              className={`p-3 rounded-lg border text-left transition ${goal === v ? 'border-blue bg-blue/20' : 'border-gray/40 hover:bg-gray/10'}`}>
              <div className={`font-semibold ${goal === v ? 'text-blue' : 'text-fg'}`}>{l}</div>
              <div className="text-xs text-gray mt-1">{hint}</div>
            </button>
          ))}
        </div>

        <label className="flex items-center gap-2 text-sm text-gray mb-6 cursor-pointer">
          <input type="checkbox" checked={useBaseline} onChange={e => setUseBaseline(e.target.checked)} className="accent-green" />
          以我当前配置（draft-mtp / q4_0 / batch4096）作为基线对比
        </label>

        <button onClick={start} disabled={state === 'running' || !selected}
          className="w-full bg-green text-bg font-bold py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition">
          {state === 'running' ? '调优进行中...' : '开始两阶段调优'}
        </button>
        {error && <div className="mt-3 text-sm text-red inline-flex items-center gap-1.5"><IconAlert size={14} />{error}</div>}
      </div>

      <div className="lg:col-span-2">
        <ProgressPanel title="测试进度" logs={logs} running={state === 'running'}
          emptyHint="配置参数后点击「开始两阶段调优」，进度将在此实时显示" />
      </div>
      </div>

      {best && (
        <div className="bg-card rounded-xl p-5 border border-green/40 mb-6">
          <div className="text-sm font-semibold text-green mb-3 inline-flex items-center gap-1.5"><IconStar size={15} />推荐配置</div>
          <div className="font-mono text-xs text-fg mb-2">{best.label}</div>
          <div className="text-xs text-gray">
            解码 {best.metrics?.decode} t/s · 预填充 {best.metrics?.prefill} t/s ·
            首字 {best.metrics?.ttft_ms} ms · GPU {best.metrics?.gpu_util}%
          </div>
          {gain != null && baseline && (
            <div className={`mt-3 text-sm font-semibold ${Number(gain) >= 0 ? 'text-green' : 'text-red'}`}>
              相比你的当前配置（{baseline.metrics?.decode} t/s），解码速度{Number(gain) >= 0 ? '提升' : '下降'} {Math.abs(Number(gain))}%
            </div>
          )}
        </div>
      )}

      {results.length > 0 && (
        <div className="bg-card rounded-xl p-4 border border-gray/30">
          <div className="text-sm font-semibold mb-3">全部测试记录</div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray border-b border-gray/30">
                <th className="text-left py-2 px-2">配置</th>
                <th className="text-right py-2 px-2">解码</th>
                <th className="text-right py-2 px-2">预填充</th>
                <th className="text-right py-2 px-2">首字ms</th>
                <th className="text-right py-2 px-2">GPU%</th>
                <th className="text-center py-2 px-2">推荐</th>
              </tr>
            </thead>
            <tbody>
              {[...results].sort((a, b) => (b.score || 0) - (a.score || 0)).map((r, i) => (
                <tr key={i} className={`border-b border-gray/20 ${r.recommended ? 'bg-green/10' : ''}`}>
                  <td className="py-2 px-2 font-mono">{r.label}</td>
                  <td className="py-2 px-2 text-right text-green">{r.metrics?.decode}</td>
                  <td className="py-2 px-2 text-right">{r.metrics?.prefill}</td>
                  <td className="py-2 px-2 text-right">{r.metrics?.ttft_ms}</td>
                  <td className="py-2 px-2 text-right">{r.metrics?.gpu_util}</td>
                  <td className="py-2 px-2 text-center">{r.recommended && <span className="text-green inline-flex justify-center"><IconStar size={14} /></span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ==================== AI 调优 ====================

function AITune({ targetId }) {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState('')
  const [ctxSize, setCtxSize] = useState(8192)
  const [goal, setGoal] = useState('latency')
  const [userDesc, setUserDesc] = useState('')

  // 配置
  const [cfg, setCfg] = useState({ api_url: '', api_key: '', model_name: '' })
  const [cfgSaved, setCfgSaved] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)
  const [showConfig, setShowConfig] = useState(false)

  // 任务
  const [state, setState] = useState('idle')
  const [logs, setLogs] = useState([])
  const [rounds, setRounds] = useState([])
  const [best, setBest] = useState(null)
  const [error, setError] = useState('')
  const pollRef = useRef(null)

  function resumePoll(jobId) {
    clearInterval(pollRef.current)
    setState('running')
    pollRef.current = setInterval(async () => {
      const pr = await fetch(`/api/ai-tune/status/${jobId}`)
      const job = await pr.json()
      setLogs(job.logs || [])
      setRounds(job.rounds || [])
      if (job.status === 'success' || job.status === 'failed') {
        clearInterval(pollRef.current)
        setBest(job.best || null)
        setState(job.status)
        if (job.error) setError(job.error)
      }
    }, 3000)
  }

  useEffect(() => {
    if (!targetId) return
    fetch(`/api/store/downloaded?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        const list = (d.models || []).map(m => m.filename)
        setModels(list)
        setSelected(list[0] || '')
      })
    // 加载已有配置
    fetch('/api/ai-tune/config').then(r => r.json()).then(d => {
      if (d.api_url) { setCfg(d); setCfgSaved(true) }
    })
    // 刷新/重进页面后恢复正在运行的 AI 调优任务
    fetch(`/api/ai-tune/active?target_id=${targetId}`)
      .then(r => r.json())
      .then(d => {
        const jobs = d.jobs || []
        if (jobs.length > 0) {
          const j = jobs[0]
          if (j.model) setSelected(j.model)
          if (j.ctx_size) setCtxSize(j.ctx_size)
          if (j.goal) setGoal(j.goal)
          setLogs(j.last_logs || [])
          resumePoll(j.job_id)
        }
      })
      .catch(() => {})
    return () => clearInterval(pollRef.current)
  }, [targetId])

  async function saveConfig() {
    await fetch('/api/ai-tune/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    })
    setCfgSaved(true)
    setShowConfig(false)
  }

  async function testConn() {
    setTesting(true); setTestResult(null)
    const res = await fetch('/api/ai-tune/test-connection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    })
    const d = await res.json()
    setTestResult(d)
    setTesting(false)
  }

  async function start() {
    setState('running'); setLogs([]); setRounds([]); setBest(null); setError('')
    const res = await fetch('/api/ai-tune/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_id: targetId, model: selected,
        ctx_size: ctxSize, goal, user_desc: userDesc,
      }),
    })
    const d = await res.json()
    if (!d.ok) { setState('failed'); setError(d.message || '启动失败'); return }
    resumePoll(d.job_id)
  }

  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-6">
      {/* 配置区 */}
      <div className="lg:col-span-3 space-y-6">
      <div className="bg-card rounded-xl p-6 border border-gray/30">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold">AI 调优配置</h2>
          <button onClick={() => setShowConfig(!showConfig)}
            className="text-sm text-blue hover:underline">
            {showConfig ? '收起' : (cfgSaved ? '修改配置' : '配置 API')}
          </button>
        </div>

        {cfgSaved && !showConfig && (
          <div className="text-sm text-green mb-2 inline-flex items-center gap-1.5"><IconCheck size={14} />已配置：{cfg.api_url} / {cfg.model_name}</div>
        )}

        {(showConfig || !cfgSaved) && (
          <div className="space-y-3">
            <div>
              <label className="block text-gray text-xs mb-1">API 地址（OpenAI 兼容）</label>
              <input value={cfg.api_url} onChange={e => setCfg({ ...cfg, api_url: e.target.value })}
                placeholder="http://192.168.50.223:8989/v1 或 https://api.deepseek.com/v1"
                className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-sm text-fg" />
            </div>
            <div>
              <label className="block text-gray text-xs mb-1">API Key（本地服务可留空）</label>
              <input value={cfg.api_key} onChange={e => setCfg({ ...cfg, api_key: e.target.value })}
                placeholder="sk-..." type="password"
                className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-sm text-fg" />
            </div>
            <div>
              <label className="block text-gray text-xs mb-1">模型名称</label>
              <input value={cfg.model_name} onChange={e => setCfg({ ...cfg, model_name: e.target.value })}
                placeholder="qwen3-27b / deepseek-chat"
                className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-sm text-fg" />
            </div>
            <div className="flex gap-3">
              <button onClick={testConn} disabled={testing || !cfg.api_url}
                className="px-4 py-2 text-sm bg-blue/20 text-blue rounded-lg disabled:opacity-40 hover:bg-blue/30 transition">
                {testing ? '测试中...' : '测试连接'}
              </button>
              <button onClick={saveConfig} disabled={!cfg.api_url}
                className="px-4 py-2 text-sm bg-green text-bg rounded-lg disabled:opacity-40 hover:opacity-90 transition">
                保存配置
              </button>
            </div>
            {testResult && (
              <div className={`text-sm inline-flex items-center gap-1.5 ${testResult.ok ? 'text-green' : 'text-red'}`}>
                {testResult.ok ? <IconCheck size={14} /> : <IconX size={14} />}{testResult.message}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 调优参数 */}
      <div className="bg-card rounded-xl p-6 border border-gray/30">
        <label className="block text-gray text-sm mb-2">选择模型</label>
        <select value={selected} onChange={e => setSelected(e.target.value)}
          className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 mb-4 text-fg">
          {models.length === 0 && <option value="">（模型目录为空）</option>}
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>

        <label className="block text-gray text-sm mb-2">上下文长度（固定约束，AI 调优不会改动它）</label>
        <CtxPicker value={ctxSize} onChange={setCtxSize} />

        <label className="block text-gray text-sm mb-2">优化目标</label>
        <div className="grid grid-cols-3 gap-3 mb-4">
          {GOALS.map(([v, l, hint]) => (
            <button key={v} onClick={() => setGoal(v)}
              className={`p-3 rounded-lg border text-left transition ${goal === v ? 'border-blue bg-blue/20' : 'border-gray/40 hover:bg-gray/10'}`}>
              <div className={`font-semibold ${goal === v ? 'text-blue' : 'text-fg'}`}>{l}</div>
              <div className="text-xs text-gray mt-1">{hint}</div>
            </button>
          ))}
        </div>

        <label className="block text-gray text-sm mb-2">场景描述（可选，帮助 AI 理解你的需求）</label>
        <textarea value={userDesc} onChange={e => setUserDesc(e.target.value)}
          placeholder="例如：主要跑长文本对话，显存比较紧张，希望首字延迟低"
          rows={2}
          className="w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-sm text-fg mb-4 resize-none" />

        <button onClick={start} disabled={state === 'running' || !selected || !cfgSaved}
          className="w-full bg-purple text-white font-bold py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition">
          {state === 'running' ? 'AI 分析中...' : '开始 AI 调优'}
        </button>
        {!cfgSaved && <div className="mt-2 text-xs text-gray">请先配置 AI API</div>}
        {error && <div className="mt-3 text-sm text-red inline-flex items-center gap-1.5"><IconAlert size={14} />{error}</div>}
      </div>
      </div>

      {/* 右栏：进度 */}
      <div className="lg:col-span-2">
        <ProgressPanel title="AI 调优进度" logs={logs} running={state === 'running'}
          emptyHint="配置 API 并点击「开始 AI 调优」，进度将在此实时显示" />
      </div>
      </div>

      {/* 每轮结果 */}
      {rounds.length > 0 && (
        <div className="bg-card rounded-xl p-4 border border-gray/30 mb-6">
          <div className="text-sm font-semibold mb-3">迭代记录（{rounds.length} 轮）</div>
          <div className="space-y-3">
            {rounds.map((r, i) => (
              <div key={i} className="bg-bg rounded-lg p-3 text-xs">
                <div className="text-gray mb-1">第 {r.round} 轮 · {r.reasoning?.slice(0, 100)}</div>
                {r.metrics ? (
                  <div className="flex gap-4 text-fg/80">
                    <span>解码 <b className="text-green">{r.metrics.decode}</b> t/s</span>
                    <span>预填充 {r.metrics.prefill} t/s</span>
                    <span>GPU {r.metrics.gpu_util}%</span>
                    <span>显存 {r.metrics.gpu_mem_pct}%</span>
                  </div>
                ) : (
                  <div className="text-red">测试失败</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 最终推荐 */}
      {best && (
        <div className="bg-card rounded-xl p-5 border border-purple/40">
          <div className="text-sm font-semibold text-purple mb-3 inline-flex items-center gap-1.5">
            <IconBot size={15} />AI 最终推荐（第 {best.round} 轮，置信度: {best.confidence}）
          </div>
          <div className="bg-bg rounded-lg p-3 font-mono text-xs text-fg mb-3 whitespace-pre-wrap">
            {Object.entries(best.params || {}).map(([k, v]) => (
              <div key={k}>--{k}{v ? ` ${v}` : ''}</div>
            ))}
          </div>
          <div className="text-xs text-gray">{best.reasoning}</div>
        </div>
      )}
    </div>
  )
}

// ==================== 主组件 ====================

export default function Tune({ targetId }) {
  const [tab, setTab] = useState('auto') // auto | ai

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">智能调优</h1>

      {/* Tab 切换 */}
      <div className="flex gap-1 mb-6 bg-card rounded-lg p-1 border border-gray/30 max-w-xs">
        <button onClick={() => setTab('auto')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-semibold transition ${tab === 'auto' ? 'bg-green text-bg' : 'text-gray hover:text-fg'}`}>
          自动调优
        </button>
        <button onClick={() => setTab('ai')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-semibold transition ${tab === 'ai' ? 'bg-purple text-white' : 'text-gray hover:text-fg'}`}>
          AI 调优
        </button>
      </div>

      {tab === 'auto' ? <AutoTune targetId={targetId} /> : <AITune targetId={targetId} />}
    </div>
  )
}
