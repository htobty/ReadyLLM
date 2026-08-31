import { useState, useEffect, useRef } from 'react'
import { IconCheck, IconX, IconAlert, IconDownload, IconHeart, IconRefresh } from '../components/Icons'

const SOURCES = [
  ['hf-mirror', 'HF 镜像站', '国内推荐'],
  ['modelscope', '魔搭', '国内最快'],
  ['huggingface', 'HuggingFace', '境外源'],
]

function fmtMB(bytes) {
  return (bytes / 1024 / 1024).toFixed(1)
}

function fmtGB(gb) {
  if (gb == null) return '?'
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(gb * 1024).toFixed(0)} MB`
}

function ModelCard({ model, targetId, downloadedMap, onDownloaded, source, activeJob, onJobStarted }) {
  const [state, setState] = useState(activeJob ? 'downloading' : 'idle')
  const [percent, setPercent] = useState(0)
  const [detail, setDetail] = useState('')
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState([])
  const pollRef = useRef(null)

  useEffect(() => () => clearInterval(pollRef.current), [])

  // 页面刷新后恢复活跃下载的轮询
  useEffect(() => {
    if (activeJob && activeJob.status === 'downloading') {
      setState('downloading')
      setShowLogs(true)
      const jobId = activeJob.job_id
      pollRef.current = setInterval(async () => {
        try {
          const pr = await fetch(`/api/store/download/${jobId}`)
          const job = await pr.json()
          setLogs(job.logs || [])
          setPercent(job.percent || 0)
          setDetail(`${fmtMB(job.downloaded || 0)} / ${job.total ? fmtMB(job.total) + ' MB' : '?'}`)
          if (job.status === 'success') {
            clearInterval(pollRef.current)
            setState('success')
            if (onDownloaded) onDownloaded()
          } else if (job.status === 'failed') {
            clearInterval(pollRef.current)
            setState('failed')
            setDetail(job.error || '下载失败')
          }
        } catch { /* ignore */ }
      }, 2000)
    }
  }, [activeJob?.job_id])

  // downloadedMap: { filename: { size_gb, expected_gb, complete } }
  const fileInfo = downloadedMap[model.filename]
  const isComplete = fileInfo && fileInfo.complete
  const isIncomplete = fileInfo && !fileInfo.complete
  const incompletePercent = fileInfo && fileInfo.expected_gb
    ? Math.min(100, Math.round((fileInfo.size_gb / fileInfo.expected_gb) * 100))
    : 0

  // 魔搭源但模型无魔搭仓库时，实际回退镜像站
  const effectiveSource =
    source === 'modelscope' && !(model.available_sources || []).includes('modelscope')
      ? 'hf-mirror'
      : source
  const srcLabel = SOURCES.find(s => s[0] === effectiveSource)?.[1] || effectiveSource

  async function startDownload() {
    setState('downloading')
    setPercent(0)
    setDetail('准备下载...')
    setShowLogs(true)
    const res = await fetch('/api/store/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: targetId, model_id: model.id, source }),
    })
    const d = await res.json()
    if (!d.ok) {
      setState('failed')
      setDetail(d.message || '启动失败')
      return
    }
    const jobId = d.job_id
    // 通知父组件注册活跃任务（刷新后可恢复）
    if (onJobStarted) onJobStarted(model.id, { job_id: jobId, model_id: model.id, status: 'downloading' })
    pollRef.current = setInterval(async () => {
      const pr = await fetch(`/api/store/download/${jobId}`)
      const job = await pr.json()
      setLogs(job.logs || [])
      setPercent(job.percent || 0)
      setDetail(`${fmtMB(job.downloaded || 0)} / ${job.total ? fmtMB(job.total) + ' MB' : '?'}`)
      if (job.status === 'success') {
        clearInterval(pollRef.current)
        setState('success')
        if (onDownloaded) onDownloaded()
      } else if (job.status === 'failed') {
        clearInterval(pollRef.current)
        setState('failed')
        setDetail(job.error || '下载失败')
      }
    }, 2000)
  }

  const fits = model.fits === false

  // 右侧状态区域
  function renderStatus() {
    if (state === 'downloading') {
      return (
        <div className="w-32">
          <div className="text-xs text-blue mb-1">{percent}%</div>
          <div className="h-1.5 bg-gray/30 rounded-full overflow-hidden">
            <div className="h-full bg-blue transition-all" style={{ width: `${percent}%` }} />
          </div>
          <div className="text-xs text-gray/60 mt-1">{detail}</div>
        </div>
      )
    }
    if (state === 'success') {
      return <span className="text-xs text-green px-3 py-2 inline-flex items-center gap-1"><IconCheck size={13} />下载完成</span>
    }
    if (isComplete) {
      return <span className="text-xs text-green px-3 py-2 inline-flex items-center gap-1"><IconCheck size={13} />已下载 {fmtGB(fileInfo.size_gb)}</span>
    }
    if (isIncomplete) {
      return (
        <div className="w-32 text-right">
          <div className="text-xs text-orange mb-1 inline-flex items-center gap-1 justify-end"><IconAlert size={12} />不完整 {incompletePercent}%</div>
          <div className="h-1.5 bg-gray/30 rounded-full overflow-hidden mb-1">
            <div className="h-full bg-orange transition-all" style={{ width: `${incompletePercent}%` }} />
          </div>
          <button
            onClick={startDownload}
            className="text-xs text-blue underline hover:opacity-80"
          >
            继续下载
          </button>
        </div>
      )
    }
    return (
      <button
        onClick={startDownload}
        disabled={fits}
        className="bg-blue text-bg text-sm font-semibold px-4 py-2 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-90 transition"
      >
        {fits ? '显存不足' : `下载·${srcLabel}`}
      </button>
    )
  }

  return (
    <div className="bg-card rounded-xl p-4 border border-gray/30">
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0">
          <div className="font-bold text-lg flex items-center gap-2">
            {model.name}
            <span className="text-xs font-normal px-2 py-0.5 rounded bg-blue/20 text-blue">{model.quant}</span>
          </div>
          <div className="text-gray text-sm mt-1">{model.desc}</div>
          <div className="text-xs text-gray/70 mt-2">
            约 {model.size_gb} GB · 需 ≥{model.min_vram_gb}G 显存
            {model.fits != null && (
              <span className={`ml-2 inline-flex items-center gap-1 ${model.fits ? 'text-green' : 'text-red'}`}>
                {model.fits ? <><IconCheck size={13} />当前机器可跑</> : <><IconX size={13} />显存不足</>}
              </span>
            )}
          </div>
        </div>
        <div className="ml-3 shrink-0">{renderStatus()}</div>
      </div>

      {state === 'failed' && detail && (
        <div className="text-xs mt-2 text-red">{detail}</div>
      )}

      {showLogs && logs.length > 0 && (
        <div className="mt-2 bg-bg rounded-lg p-2 max-h-28 overflow-auto font-mono text-xs text-fg/70 space-y-0.5">
          {logs.map((l, i) => (
            <div key={i}><span className="text-gray/50">[{l.t}]</span> {l.msg}</div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Store({ targetId }) {
  const [models, setModels] = useState([])
  const [dynamicModels, setDynamicModels] = useState([])
  const [downloadedMap, setDownloadedMap] = useState({})
  const [activeJobs, setActiveJobs] = useState({}) // { model_id: job }
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [refreshMsg, setRefreshMsg] = useState('')
  const [filter, setFilter] = useState('all') // all | fits
  const [source, setSource] = useState('hf-mirror') // hf-mirror | modelscope | huggingface
  const [tab, setTab] = useState('curated') // curated | dynamic

  async function loadDownloaded() {
    if (!targetId) return
    const res = await fetch(`/api/store/downloaded?target_id=${targetId}`)
    const d = await res.json()
    // 构建 { filename: { size_gb, expected_gb, complete } } map
    const map = {}
    for (const m of (d.models || [])) {
      map[m.filename] = {
        size_gb: m.size_gb,
        expected_gb: m.expected_gb,
        complete: m.complete,
      }
    }
    setDownloadedMap(map)
  }

  // 页面加载时恢复活跃下载任务（刷新后不丢失进度）
  async function loadActiveJobs() {
    if (!targetId) return
    try {
      const res = await fetch(`/api/store/jobs?target_id=${targetId}`)
      const d = await res.json()
      const map = {}
      for (const j of (d.jobs || [])) {
        if (j.status === 'downloading') {
          map[j.model_id] = j
        }
      }
      setActiveJobs(map)
    } catch { /* ignore */ }
  }

  async function loadDynamic() {
    try {
      const res = await fetch('/api/store/dynamic')
      const d = await res.json()
      setDynamicModels(d.models || [])
    } catch { /* ignore */ }
  }

  async function handleRefresh() {
    setRefreshing(true)
    setRefreshMsg('正在从 HuggingFace 获取最新模型...')
    try {
      const res = await fetch('/api/store/refresh')
      const d = await res.json()
      setDynamicModels(d.models || [])
      if (d.error) {
        setRefreshMsg(`${d.error}`)
      } else {
        setRefreshMsg(`已更新，获取到 ${d.models?.length || 0} 个热门模型`)
      }
      setTab('dynamic')
    } catch (e) {
      setRefreshMsg(`刷新失败: ${e.message}`)
    } finally {
      setRefreshing(false)
      setTimeout(() => setRefreshMsg(''), 5000)
    }
  }

  useEffect(() => {
    if (!targetId) return
    setLoading(true)
    fetch(`/api/store/models?target_id=${targetId}&source=${source}`)
      .then(r => r.json())
      .then(d => {
        setModels(d.models || [])
        setLoading(false)
      })
    loadDownloaded()
    loadDynamic()
    loadActiveJobs()
  }, [targetId, source])

  const shown = filter === 'fits' ? models.filter(m => m.fits !== false) : models

  return (
    <div>
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h1 className="text-2xl font-bold">模型商店</h1>
        <div className="flex gap-2 items-center">
          {[['all', '全部'], ['fits', '可跑的']].map(([v, l]) => (
            <button
              key={v}
              onClick={() => setFilter(v)}
              className={`px-3 py-1.5 rounded-lg text-sm border transition ${
                filter === v ? 'border-blue bg-blue/20 text-blue' : 'border-gray/40 text-fg/70'
              }`}
            >
              {l}
            </button>
          ))}
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="px-3 py-1.5 rounded-lg text-sm border border-purple/50 text-purple hover:bg-purple/10 transition disabled:opacity-50 flex items-center gap-1"
          >
            <span className={refreshing ? 'animate-spin inline-block' : ''}>⟳</span>
            {refreshing ? '刷新中...' : '刷新最新'}
          </button>
        </div>
      </div>

      {refreshMsg && (
        <div className="text-sm text-gray mb-3 px-3 py-2 bg-card rounded-lg border border-gray/20">{refreshMsg}</div>
      )}

      {/* Tab 切换：精选 / 热门动态 */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => setTab('curated')}
          className={`px-4 py-2 rounded-lg text-sm font-medium border transition ${
            tab === 'curated' ? 'border-blue bg-blue/20 text-blue' : 'border-gray/40 text-fg/70 hover:bg-gray/10'
          }`}
        >
          精选推荐
        </button>
        <button
          onClick={() => setTab('dynamic')}
          className={`px-4 py-2 rounded-lg text-sm font-medium border transition ${
            tab === 'dynamic' ? 'border-purple bg-purple/20 text-purple' : 'border-gray/40 text-fg/70 hover:bg-gray/10'
          }`}
        >
          热门动态 {dynamicModels.length > 0 && <span className="text-xs opacity-70">({dynamicModels.length})</span>}
        </button>
      </div>

      {/* 下载源选择器（仅精选模式） */}
      {tab === 'curated' && (
        <div className="flex items-center gap-2 mb-6 flex-wrap">
          <span className="text-sm text-gray">下载源：</span>
          {SOURCES.map(([v, l, hint]) => (
            <button
              key={v}
              onClick={() => setSource(v)}
              className={`px-3 py-1.5 rounded-lg text-sm border transition ${
                source === v ? 'border-green bg-green/20 text-green' : 'border-gray/40 text-fg/70 hover:bg-gray/10'
              }`}
              title={hint}
            >
              {l}
              <span className="text-xs opacity-60 ml-1">· {hint}</span>
            </button>
          ))}
        </div>
      )}

      {tab === 'curated' ? (
        loading ? (
          <div className="text-gray">加载模型列表...</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {shown.map(m => (
              <ModelCard
                key={m.id}
                model={m}
                targetId={targetId}
                source={source}
                downloadedMap={downloadedMap}
                onDownloaded={loadDownloaded}
                activeJob={activeJobs[m.id] || null}
                onJobStarted={(modelId, job) => setActiveJobs(prev => ({ ...prev, [modelId]: job }))}
              />
            ))}
          </div>
        )
      ) : (
        <div>
          {dynamicModels.length === 0 ? (
            <div className="text-gray py-8 text-center">
              暂无动态数据，点击「刷新最新」从 HuggingFace 获取热门模型
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {dynamicModels.map(m => (
                <div key={m.id} className="bg-card rounded-xl p-4 border border-gray/30">
                  <div className="flex justify-between items-start">
                    <div className="flex-1 min-w-0">
                      <div className="font-bold text-base truncate">{m.name}</div>
                      <div className="text-xs text-gray mt-1 flex items-center gap-3">
                        <span className="inline-flex items-center gap-1"><IconDownload size={12} />{(m.downloads / 1000).toFixed(0)}K</span>
                        <span className="inline-flex items-center gap-1"><IconHeart size={12} />{m.likes}</span>
                      </div>
                      <div className="text-xs text-gray/60 mt-1 truncate">{m.repo}</div>
                    </div>
                    <a
                      href={m.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-3 shrink-0 text-sm px-3 py-1.5 rounded-lg border border-purple/50 text-purple hover:bg-purple/10 transition"
                    >
                      查看仓库
                    </a>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
