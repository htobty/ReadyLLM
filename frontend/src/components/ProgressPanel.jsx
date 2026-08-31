// 可复用进度面板：标题 + 实时日志列表，用于调优页右侧栏
// running=true 时显示活动指示；logs 为空时显示等待占位

export default function ProgressPanel({ title = '调优进度', logs = [], running = false, emptyHint = '等待任务启动…' }) {
  return (
    <div className="bg-card rounded-xl p-4 border border-gray/30 lg:sticky lg:top-6">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-semibold">{title}</span>
        {running && (
          <span className="flex items-center gap-1 text-xs text-green">
            <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse" />
            运行中
          </span>
        )}
      </div>
      <div className="bg-bg rounded-lg p-3 h-[60vh] max-h-[640px] min-h-[280px] overflow-auto font-mono text-xs text-fg/80 space-y-0.5">
        {logs.length === 0
          ? <div className="text-gray/50">{emptyHint}</div>
          : logs.map((l, i) => (
              <div key={i}><span className="text-gray/50">[{l.t}]</span> {l.msg}</div>
            ))}
      </div>
    </div>
  )
}
