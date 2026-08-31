import { useWebSocket } from '../hooks/useWebSocket'
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { useRef } from 'react'

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
  const historyRef = useRef({ speed: [], cache: [], spec: [] })
  const lastRef = useRef({ completion_tokens: 0, prompt_tokens: 0 })

  // 切换目标机器时清空历史
  const curTargetRef = useRef(targetId)
  if (curTargetRef.current !== targetId) {
    curTargetRef.current = targetId
    historyRef.current = { speed: [], cache: [], spec: [] }
    lastRef.current = { completion_tokens: 0, prompt_tokens: 0 }
  }

  // 只在累计值变化时追加描点
  if (data?.metrics) {
    const m = data.metrics
    const h = historyRef.current
    const l = lastRef.current

    if (m.completion_tokens !== l.completion_tokens && m.completion_speed > 0) {
      h.speed.push({ idx: h.speed.length, value: m.completion_speed })
      if (h.speed.length > MAX_POINTS) h.speed.shift()
    }
    if (m.prompt_tokens !== l.prompt_tokens && m.cache_hit_rate > 0) {
      h.cache.push({ idx: h.cache.length, value: m.cache_hit_rate })
      if (h.cache.length > MAX_POINTS) h.cache.shift()
    }
    if (m.spec_accept_rate > 0) {
      h.spec.push({ idx: h.spec.length, value: m.spec_accept_rate })
      if (h.spec.length > MAX_POINTS) h.spec.shift()
    }

    lastRef.current = {
      completion_tokens: m.completion_tokens,
      prompt_tokens: m.prompt_tokens,
    }
  }

  const h = historyRef.current
  const gpu = data?.gpu || {}
  const cpuMem = data?.cpu_mem || {}
  const metrics = data?.metrics || {}

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">实时监控</h1>
        <div className={`flex items-center gap-2 text-sm ${connected ? 'text-green' : 'text-red'}`}>
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green animate-pulse' : 'bg-red'}`} />
          {connected ? '已连接' : '未连接'}
        </div>
      </div>

      {/* GPU + CPU 并排 */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-card rounded-xl p-4 border border-gray/30">
          <div className="text-sm font-semibold text-yellow mb-3">GPU</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="col-span-2 truncate"><span className="text-gray">型号:</span> {gpu.name || '--'}</div>
            <div><span className="text-gray">利用率:</span> {gpu.utilization != null ? `${gpu.utilization}%` : '--'}</div>
            <div><span className="text-gray">温度:</span> {gpu.temperature ? `${gpu.temperature}°C` : '--'}</div>
            <div><span className="text-gray">显存:</span> {gpu.memory_used_gb != null ? `${gpu.memory_used_gb}G / ${gpu.memory_total_gb}G` : '--'}</div>
            <div><span className="text-gray">功耗:</span> {gpu.power ? `${gpu.power}W` : '--'}</div>
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
          <div className="text-sm font-semibold text-green mb-3">CPU / 内存</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><span className="text-gray">CPU:</span> {cpuMem.cpu_pct != null ? `${cpuMem.cpu_pct}%` : '--'}</div>
            <div><span className="text-gray">内存:</span> {cpuMem.memory_used_gb != null ? `${cpuMem.memory_used_gb}G / ${cpuMem.memory_total_gb}G` : '--'}</div>
          </div>
          <div className="mt-2 space-y-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-gray w-8">CPU</span>
              <div className="flex-1 h-2 bg-gray/30 rounded-full overflow-hidden">
                <div className="h-full bg-green rounded-full transition-all" style={{ width: `${cpuMem.cpu_pct || 0}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-gray w-8">内存</span>
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
        <MetricCard label="生成 Tokens" value={metrics.completion_tokens} color="text-green" />
        <MetricCard label="生成速度" value={metrics.completion_speed} unit="t/s" color="text-green" />
        <MetricCard label="Prompt 速度" value={metrics.prompt_speed} unit="t/s" color="text-blue" />
        <MetricCard label="缓存命中率" value={metrics.cache_hit_rate} unit="%" color="text-purple" />
        <MetricCard label="投机接受率" value={metrics.spec_accept_rate} unit="%" color="text-teal" />
      </div>

      {/* 曲线图 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartPanel title="Token 生成速度 (t/s)" data={h.speed} dataKey="value" color="#a6e3a1" unit=" t/s" />
        <ChartPanel title="缓存命中率 (%)" data={h.cache} dataKey="value" color="#cba6f7" unit="%" />
        <ChartPanel title="投机采样接受率 (%)" data={h.spec} dataKey="value" color="#94e2d5" unit="%" />
      </div>
    </div>
  )
}
