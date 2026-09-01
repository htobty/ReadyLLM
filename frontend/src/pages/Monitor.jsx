import { useWebSocket } from '../hooks/useWebSocket'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { useState, useEffect } from 'react'
import { useI18n } from '../i18n/I18nContext'

const MAX_POINTS = 60

function MetricCard({ label, value, unit, color }) {
  return (
    <div className="bg-card rounded-lg p-3 border border-gray/30">
      <div className="text-gray text-xs mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>
        {value ?? '--'}
        {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
      </div>
    </div>
  )
}

function ChartPanel({ title, data, dataKey, color, unit }) {
  return (
    <div className="bg-card rounded-lg p-4 border border-gray/30">
      <div className="text-sm font-semibold mb-2">{title}</div>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id={`grad-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#3a3a4c" />
          <XAxis dataKey="idx" hide />
          <YAxis stroke="#6c7086" fontSize={10} tickLine={false} width={35} />
          <Tooltip
            contentStyle={{ background: '#181825', border: '1px solid #45475a', borderRadius: 8, fontSize: 12 }}
            labelFormatter={() => ''}
            formatter={(val) => [`${val}${unit}`, title]}
          />
          <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#grad-${dataKey})`} strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Monitor({ targetId }) {
  const { data, connected } = useWebSocket(
    targetId ? `ws://localhost:8000/api/monitor/ws?target_id=${targetId}` : ''
  )
  const { t } = useI18n()
  const [history, setHistory] = useState({ speed: [], cache: [], spec: [] })

  // 切换目标机器时清空历史
  useEffect(() => {
    setHistory({ speed: [], cache: [], spec: [] })
  }, [targetId])

  // 每 10s 采集一次；值有变化才追加描点，没变化不描
  useEffect(() => {
    if (!data?.metrics) return
    const m = data.metrics
    setHistory((prev) => {
      const withPoint = (arr, value) => {
        if (arr.length > 0 && arr[arr.length - 1].value === value) return arr
        return [...arr, { idx: arr.length, value }].slice(-MAX_POINTS)
      }
      const speed = withPoint(prev.speed, m.completion_speed)
      const cache = withPoint(prev.cache, m.cache_hit_rate)
      const spec = withPoint(prev.spec, m.spec_accept_rate)
      if (speed === prev.speed && cache === prev.cache && spec === prev.spec) return prev
      return { speed, cache, spec }
    })
  }, [data])

  const h = history
  const gpu = data?.gpu || {}
  const cpuMem = data?.cpu_mem || {}
  const metrics = data?.metrics || {}

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">{t('monitor.title')}</h1>
        <div className={`flex items-center gap-2 text-sm ${connected ? 'text-green' : 'text-red'}`}>
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green animate-pulse' : 'bg-red'}`} />
          {connected ? t('monitor.connected') : t('monitor.disconnected')}
        </div>
      </div>

      {/* GPU + CPU 并排 */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-card rounded-xl p-4 border border-gray/30">
          <div className="text-sm font-semibold text-yellow mb-3">GPU</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="col-span-2 truncate"><span className="text-gray">{t('monitor.model')}</span> {gpu.name || '--'}</div>
            <div><span className="text-gray">{t('monitor.utilization')}</span> {gpu.utilization != null ? `${gpu.utilization}%` : '--'}</div>
            <div><span className="text-gray">{t('monitor.temp')}</span> {gpu.temperature ? `${gpu.temperature}°C` : '--'}</div>
            <div><span className="text-gray">{t('monitor.vram')}</span> {gpu.memory_used_gb != null ? `${gpu.memory_used_gb}G / ${gpu.memory_total_gb}G` : '--'}</div>
            <div><span className="text-gray">{t('monitor.power')}</span> {gpu.power ? `${gpu.power}W` : '--'}</div>
          </div>
          {gpu.memory_pct != null && (
            <div className="mt-2">
              <div className="h-2 bg-gray/30 rounded-full overflow-hidden">
                <div className="h-full bg-blue rounded-full transition-all" style={{ width: `${gpu.memory_pct}%` }} />
              </div>
            </div>
          )}
        </div>

        <div className="bg-card rounded-xl p-4 border border-gray/30">
          <div className="text-sm font-semibold text-green mb-3">{t('monitor.cpuMem')}</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><span className="text-gray">{t('monitor.cpu')}</span> {cpuMem.cpu_pct != null ? `${cpuMem.cpu_pct}%` : '--'}</div>
            <div><span className="text-gray">{t('monitor.memory')}</span> {cpuMem.memory_used_gb != null ? `${cpuMem.memory_used_gb}G / ${cpuMem.memory_total_gb}G` : '--'}</div>
          </div>
          <div className="mt-2 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-gray w-8">CPU</span>
              <div className="flex-1 h-2 bg-gray/30 rounded-full overflow-hidden">
                <div className="h-full bg-green rounded-full transition-all" style={{ width: `${cpuMem.cpu_pct || 0}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-gray w-8">{t('monitor.mem')}</span>
              <div className="flex-1 h-2 bg-gray/30 rounded-full overflow-hidden">
                <div className="h-full bg-yellow rounded-full transition-all" style={{ width: `${cpuMem.memory_pct || 0}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 推理指标数值 */}
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        <MetricCard label="Prompt Tokens" value={metrics.prompt_tokens} color="text-blue" />
        <MetricCard label={t('monitor.completionTokens')} value={metrics.completion_tokens} color="text-green" />
        <MetricCard label={t('monitor.genSpeed')} value={metrics.completion_speed} unit="t/s" color="text-green" />
        <MetricCard label={t('monitor.promptSpeed')} value={metrics.prompt_speed} unit="t/s" color="text-blue" />
        <MetricCard label={t('monitor.cacheHit')} value={metrics.cache_hit_rate} unit="%" color="text-purple" />
        <MetricCard label={t('monitor.specAccept')} value={metrics.spec_accept_rate} unit="%" color="text-teal" />
      </div>

      {/* 曲线图 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartPanel title={t('monitor.chartSpeed')} data={h.speed} dataKey="value" color="#a6e3a1" unit=" t/s" />
        <ChartPanel title={t('monitor.chartCache')} data={h.cache} dataKey="value" color="#cba6f7" unit="%" />
        <ChartPanel title={t('monitor.chartSpec')} data={h.spec} dataKey="value" color="#94e2d5" unit="%" />
      </div>
    </div>
  )
}
