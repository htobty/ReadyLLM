import { useState, useEffect } from 'react'
import Monitor from './pages/Monitor'
import Deploy from './pages/Deploy'
import Settings from './pages/Settings'
import Store from './pages/Store'
import Tune from './pages/Tune'
import {
  IconActivity, IconStore, IconRocket, IconSliders, IconSettings,
  IconServer,
} from './components/Icons'
import Logo from './components/Logo'
import { I18nProvider, useI18n, LangSwitch } from './i18n/I18nContext'

// 首页直接复用实时监控视图
const NAV_ITEMS = [
  { id: 'dashboard', label: 'nav.monitor', icon: IconActivity },
  { id: 'store', label: 'nav.store', icon: IconStore },
  { id: 'deploy', label: 'nav.deploy', icon: IconRocket },
  { id: 'tune', label: 'nav.tune', icon: IconSliders },
  { id: 'settings', label: 'nav.settings', icon: IconSettings },
]

export default function App() {
  return (
    <I18nProvider>
      <AppInner />
    </I18nProvider>
  )
}

function AppInner() {
  const { t } = useI18n()
  const [page, setPage] = useState('dashboard')
  const [targets, setTargets] = useState([])
  const [targetId, setTargetId] = useState('')

  useEffect(() => {
    fetch('/api/target')
      .then(r => r.json())
      .then(d => {
        const list = d.targets || []
        setTargets(list)
        if (list.length && !list.find(t => t.id === targetId)) {
          setTargetId(list[0].id)
        }
      })
  }, [])

  function handleSaved(list) {
    setTargets(list)
    if (list.length && !list.find(t => t.id === targetId)) {
      setTargetId(list[0].id)
    }
  }

  function refreshTargets() {
    fetch('/api/target')
      .then(r => r.json())
      .then(d => setTargets(d.targets || []))
  }

  const hasTarget = targets.length > 0
  const current = targets.find(t => t.id === targetId)

  return (
    <div className="flex h-screen text-fg">
      {/* 侧边导航 */}
      <nav className="w-60 shrink-0 border-r border-white/5 bg-card/40 backdrop-blur-xl p-4 flex flex-col">
        {/* 品牌区 */}
        <div className="flex items-center gap-3 px-2 pt-1 pb-6">
          <Logo size={38} />
          <div className="leading-tight">
            <div className="font-bold text-[17px] tracking-tight bg-gradient-to-r from-blue via-purple to-teal bg-clip-text text-transparent">ReadyLLM</div>
            <div className="text-[11px] text-gray">{t('brand.sub')}</div>
          </div>
        </div>

        {/* 导航项 */}
        <div className="flex flex-col gap-1">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon
            const active = page === item.id
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                  active
                    ? 'bg-gradient-to-r from-blue/15 to-transparent text-fg font-semibold'
                    : 'text-fg/55 hover:text-fg hover:bg-white/[0.04]'
                }`}
              >
                <span
                  className={`absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-blue transition-opacity ${
                    active ? 'opacity-100' : 'opacity-0'
                  }`}
                />
                <span className={active ? 'text-blue' : 'text-fg/45 group-hover:text-fg/80'}>
                  <Icon size={18} />
                </span>
                {t(item.label)}
              </button>
            )
          })}
        </div>

        {/* 目标机器选择器 */}
        {page !== 'settings' && hasTarget && (
          <div className="mt-auto pt-4 border-t border-white/5">
            <div className="text-[11px] uppercase tracking-wider text-gray mb-2 px-2 flex items-center gap-1.5">
              <IconServer size={13} /> {t('app.targetMachine')}
            </div>
            <select
              value={targetId}
              onChange={e => setTargetId(e.target.value)}
              className="w-full bg-bg/60 border border-white/10 rounded-lg px-3 py-2 text-sm text-fg focus:border-blue/50 outline-none transition"
            >
              {targets.map(tg => (
                <option key={tg.id} value={tg.id}>
                  {tg.name} ({tg.conn_type === 'ssh' ? tg.host : t('app.local')})
                </option>
              ))}
            </select>
          </div>
        )}
        <LangSwitch />
      </nav>

      {/* 主内容 */}
      <main className="flex-1 overflow-auto p-8">
        {page === 'settings' ? (
          <Settings targets={targets} onSaved={handleSaved} onChanged={refreshTargets} />
        ) : !hasTarget ? (
          <div className="h-full flex flex-col items-center justify-center text-center animate-slide-up">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue/20 to-purple/20 border border-white/10 flex items-center justify-center text-blue mb-6 shadow-glow">
              <IconSettings size={36} />
            </div>
            <div className="text-2xl font-bold mb-3 tracking-tight">{t('app.noTarget')}</div>
            <div className="text-gray mb-8 max-w-md leading-relaxed">
              {t('app.noTargetHint')}
            </div>
            <button
              onClick={() => setPage('settings')}
              className="bg-blue text-bg font-bold px-7 py-2.5 rounded-xl hover:opacity-90 transition shadow-glow"
            >
              {t('app.goSettings')}
            </button>
          </div>
        ) : page === 'dashboard' ? (
          <Monitor targetId={targetId} />
        ) : page === 'store' ? (
          <Store targetId={targetId} />
        ) : page === 'deploy' ? (
          <Deploy targetId={targetId} target={current} />
        ) : page === 'tune' ? (
          <Tune targetId={targetId} />
        ) : (
          <Monitor targetId={targetId} />
        )}
      </main>
    </div>
  )
}
