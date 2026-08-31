import { useState, useEffect, useRef } from 'react'
import { IconAlert, IconCheck, IconX } from '../components/Icons'

const EMPTY = {
  conn_type: 'local',
  os: 'windows',
  name: '我的电脑',
  host: '',
  port: 22,
  user: '',
  auth_type: 'key',
  key_path: '',
  password: '',
  engine_type: 'llama_cpp',
  engine_path: '',
  models_dir: '',
  service_port: 8080,
}

const OS_OPTIONS = [
  ['windows', 'Windows'],
  ['macos', 'macOS'],
  ['linux', 'Linux'],
]

const OS_LABEL = { windows: 'Windows', macos: 'macOS', linux: 'Linux' }

// llama.cpp 各系统的可执行文件路径示例
function llamaPlaceholder(os) {
  if (os === 'windows') return 'C:\\llama\\llama-server.exe'
  if (os === 'macos') return '/opt/homebrew/bin/llama-server'
  return '/usr/local/bin/llama-server'
}
// 引擎路径占位提示：vLLM 用命令名，llama.cpp 用完整路径
function enginePlaceholder(engineType, os) {
  if (engineType === 'vllm') return 'vllm（pip 安装后默认在 PATH，可留空）'
  return llamaPlaceholder(os)
}
function modelsPlaceholder(os) {
  if (os === 'windows') return 'D:\\models'
  if (os === 'macos') return '~/models'
  return '/home/user/models'
}

function Field({ label, children, hint }) {
  return (
    <div className="mb-4">
      <label className="block text-sm text-gray mb-1">{label}</label>
      {children}
      {hint && <div className="text-xs text-gray/70 mt-1">{hint}</div>}
    </div>
  )
}

const inputCls =
  'w-full bg-bg border border-gray/40 rounded-lg px-3 py-2 text-fg focus:border-blue outline-none'

function SegButtons({ value, options, onChange }) {
  return (
    <div className="flex gap-3">
      {options.map(([v, l]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`flex-1 py-2 rounded-lg border transition ${
            value === v ? 'border-blue bg-blue/20 text-blue' : 'border-gray/40 text-fg/70'
          }`}
        >
          {l}
        </button>
      ))}
    </div>
  )
}

export default function Settings({ targets, onSaved, onChanged }) {
  const [form, setForm] = useState(EMPTY)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [localOs, setLocalOs] = useState('')
  const [engines, setEngines] = useState([])

  function set(k, v) {
    setForm(f => ({ ...f, [k]: v }))
  }

  const isRemote = form.conn_type === 'ssh'
  // 当前引擎元信息（来自后端单一数据源）
  const curEngine = engines.find(e => e.type === form.engine_type)
  // 该引擎是否不支持当前目标 OS（如 vLLM + Windows）
  const engineOsUnsupported =
    curEngine && curEngine.supported_os && !curEngine.supported_os.includes(form.os)

  // 拉取本机操作系统，本机模式下自动识别，无需用户手选
  useEffect(() => {
    fetch('/api/target/local-os')
      .then(r => r.json())
      .then(d => {
        if (d.os) {
          setLocalOs(d.os)
          setForm(f => (f.conn_type === 'local' ? { ...f, os: d.os } : f))
        }
      })
      .catch(() => {})
    // 拉取可用引擎元信息
    fetch('/api/target/engines')
      .then(r => r.json())
      .then(d => setEngines(d.engines || []))
      .catch(() => {})
  }, [])

  // 切换连接方式时同步 OS：本机->用识别值，远程->保留/默认
  function switchConn(v) {
    setForm(f => ({ ...f, conn_type: v, os: v === 'local' && localOs ? localOs : f.os }))
  }

  async function test() {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await fetch('/api/target/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      setTestResult(await res.json())
    } catch (e) {
      setTestResult({ ok: false, message: String(e) })
    } finally {
      setTesting(false)
    }
  }

  async function save() {
    const res = await fetch('/api/target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })
    const d = await res.json()
    if (d.ok) {
      if (onSaved) onSaved(d.targets)
      setTestResult(null)
    }
  }

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-bold mb-2">目标机器设置</h1>
      <p className="text-gray text-sm mb-6">
        配置你要部署/监控的机器：可以是本机，也可以是局域网内的其他电脑。
      </p>

      <div className="bg-card rounded-xl p-6 border border-gray/30">
        <Field label="机器名称">
          <input className={inputCls} value={form.name} onChange={e => set('name', e.target.value)} />
        </Field>

        <Field label="连接方式">
          <SegButtons value={form.conn_type} options={[['local', '本机'], ['ssh', '远程 (SSH)']]} onChange={switchConn} />
        </Field>

        <Field label="操作系统">
          {form.conn_type === 'local' ? (
            <div className="flex items-center gap-2 py-2 px-3 rounded-lg bg-bg border border-gray/40">
              <span className="w-2 h-2 rounded-full bg-green" />
              <span className="text-fg">
                {localOs ? OS_LABEL[localOs] : '自动识别中...'}
                <span className="text-gray text-xs ml-2">（本机自动检测）</span>
              </span>
            </div>
          ) : (
            <SegButtons value={form.os} options={OS_OPTIONS} onChange={v => set('os', v)} />
          )}
        </Field>

        {isRemote && (
          <div className="border-l-2 border-blue/40 pl-4 my-4">
            <Field label="主机地址 (IP)">
              <input className={inputCls} placeholder="192.168.1.100" value={form.host} onChange={e => set('host', e.target.value)} />
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="SSH 端口">
                <input type="number" className={inputCls} value={form.port} onChange={e => set('port', +e.target.value)} />
              </Field>
              <Field label="用户名">
                <input className={inputCls} value={form.user} onChange={e => set('user', e.target.value)} />
              </Field>
            </div>
            <Field label="认证方式">
              <SegButtons value={form.auth_type} options={[['key', '密钥'], ['password', '密码']]} onChange={v => set('auth_type', v)} />
            </Field>
            {form.auth_type === 'key' ? (
              <Field label="私钥路径" hint="留空则默认使用 ~/.ssh/id_rsa">
                <input className={inputCls} placeholder="~/.ssh/id_rsa" value={form.key_path} onChange={e => set('key_path', e.target.value)} />
              </Field>
            ) : (
              <Field label="密码">
                <input type="password" className={inputCls} value={form.password} onChange={e => set('password', e.target.value)} />
              </Field>
            )}
          </div>
        )}

        <Field label="推理引擎">
          <SegButtons
            value={form.engine_type}
            options={
              engines.length > 0
                ? engines.map(e => [e.type, e.label])
                : [['llama_cpp', 'llama.cpp'], ['vllm', 'vLLM']]
            }
            onChange={v => set('engine_type', v)}
          />
          {curEngine?.desc && (
            <div className="text-xs text-gray/70 mt-2">{curEngine.desc}</div>
          )}
        </Field>

        {engineOsUnsupported && (
          <div className="mb-4 p-3 rounded-lg bg-yellow/10 text-yellow text-sm border border-yellow/30 flex items-start gap-2">
            <span className="mt-0.5 shrink-0"><IconAlert size={15} /></span>
            <span>{curEngine?.windows_note || `所选引擎不支持 ${OS_LABEL[form.os]}，请改用其他引擎或调整目标系统`}</span>
          </div>
        )}

        <Field
          label={form.engine_type === 'vllm' ? 'vLLM 命令路径' : '推理引擎路径'}
          hint={
            form.engine_type === 'vllm'
              ? 'vllm 可执行命令，pip 安装后通常在 PATH，可留空（未安装可保存后用下方一键安装）'
              : 'llama-server 可执行文件的完整路径（未安装可留空，保存后用下方一键安装）'
          }
        >
          <input
            className={inputCls}
            placeholder={enginePlaceholder(form.engine_type, form.os)}
            value={form.engine_path}
            onChange={e => set('engine_path', e.target.value)}
          />
        </Field>

        <Field
          label="模型目录"
          hint={
            form.engine_type === 'vllm'
              ? '存放 HuggingFace safetensors 权重目录的父目录（vLLM 用模型 id 或本地权重目录）'
              : '存放 .gguf 模型文件的目录'
          }
        >
          <input
            className={inputCls}
            placeholder={modelsPlaceholder(form.os)}
            value={form.models_dir}
            onChange={e => set('models_dir', e.target.value)}
          />
        </Field>

        <Field label="服务端口" hint="llama-server 监听的端口">
          <input type="number" className={inputCls} value={form.service_port} onChange={e => set('service_port', +e.target.value)} />
        </Field>

        <div className="flex gap-3 mt-6">
          <button
            onClick={test}
            disabled={testing}
            className="flex-1 py-2 rounded-lg border border-blue text-blue font-semibold disabled:opacity-40 hover:bg-blue/10 transition"
          >
            {testing ? '测试中...' : '测试连接'}
          </button>
          <button
            onClick={save}
            className="flex-1 py-2 rounded-lg bg-green text-bg font-bold hover:opacity-90 transition"
          >
            保存
          </button>
        </div>

        {testResult && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${testResult.ok ? 'bg-green/10 text-green' : 'bg-red/10 text-red'}`}>
            <div className="font-semibold mb-1 inline-flex items-center gap-1.5">{testResult.ok ? <><IconCheck size={15} />连接成功</> : <><IconX size={15} />连接失败</>}</div>
            <div>{testResult.message}</div>
            {testResult.hardware?.gpu && (
              <div className="mt-2 text-fg/80">
                GPU: {testResult.hardware.gpu.name} ({testResult.hardware.gpu.total_memory_gb}G) ·
                内存: {testResult.hardware.memory?.total_gb}G
              </div>
            )}
          </div>
        )}
      </div>

      {/* 引擎安装面板 */}
      {targets?.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-2">推理引擎</h2>
          <p className="text-gray text-sm mb-4">
            检测各机器是否已安装其配置的引擎（llama.cpp / vLLM），未安装可一键安装（自动下载/编译/pip 并回填路径）。
          </p>
          <div className="space-y-3">
            {targets.map(t => (
              <EngineRow key={t.id} target={t} onChanged={onChanged} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function EngineRow({ target, onChanged }) {
  const [state, setState] = useState('checking') // checking | installed | missing | installing
  const [engine, setEngine] = useState(null)
  const [logs, setLogs] = useState([])
  const [showLogs, setShowLogs] = useState(false)
  const pollRef = useRef(null)

  async function check() {
    setState('checking')
    try {
      const res = await fetch(`/api/target/${target.id}/engine`)
      const d = await res.json()
      setEngine(d)
      setState(d.installed ? 'installed' : 'missing')
    } catch {
      setState('missing')
    }
  }

  useEffect(() => {
    check()
    return () => clearInterval(pollRef.current)
  }, [target.id])

  async function install() {
    setState('installing')
    setLogs([])
    setShowLogs(true)
    const res = await fetch('/api/target/install-engine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_id: target.id }),
    })
    const d = await res.json()
    if (!d.ok) {
      setLogs([{ t: '--:--:--', msg: d.message || '启动失败' }])
      setState('missing')
      return
    }
    const jobId = d.job_id
    pollRef.current = setInterval(async () => {
      const sr = await fetch(`/api/target/install-status/${jobId}`)
      const job = await sr.json()
      setLogs(job.logs || [])
      if (job.status === 'success') {
        clearInterval(pollRef.current)
        setEngine({ installed: true, path: job.engine_path, version: '' })
        setState('installed')
        if (onChanged) onChanged()
      } else if (job.status === 'failed') {
        clearInterval(pollRef.current)
        setState('missing')
      }
    }, 2000)
  }

  const badge = {
    checking: { text: '检测中...', cls: 'bg-gray/20 text-gray' },
    installed: { text: '已安装', cls: 'bg-green/20 text-green' },
    missing: { text: '未安装', cls: 'bg-yellow/20 text-yellow' },
    installing: { text: '安装中...', cls: 'bg-blue/20 text-blue' },
  }[state]

  return (
    <div className="bg-card rounded-xl p-4 border border-gray/30">
      <div className="flex items-center justify-between">
        <div>
          <div className="font-semibold">
            {target.name}
            <span className="ml-2 text-xs px-2 py-0.5 rounded bg-blue/15 text-blue font-normal">
              {target.engine_type === 'vllm' ? 'vLLM' : 'llama.cpp'}
            </span>
          </div>
          <div className="text-xs text-gray">
            {target.os} · {target.conn_type === 'ssh' ? target.host : '本机'}
            {engine?.version && <span className="ml-2 text-green">{engine.version}</span>}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-1 rounded-full ${badge.cls}`}>{badge.text}</span>
          {state === 'missing' && (
            <button
              onClick={install}
              className="bg-blue text-bg text-sm font-semibold px-4 py-1.5 rounded-lg hover:opacity-90 transition"
            >
              一键安装
            </button>
          )}
          {state === 'installed' && (
            <button onClick={check} className="text-gray text-sm hover:text-fg transition">重新检测</button>
          )}
        </div>
      </div>

      {engine?.installed && engine.path && (
        <div className="text-xs text-gray/70 mt-2 truncate">路径: {engine.path}</div>
      )}

      {(showLogs && logs.length > 0) && (
        <div className="mt-3">
          <button onClick={() => setShowLogs(s => !s)} className="text-xs text-gray mb-1">
            {showLogs ? '收起日志 ▲' : '查看日志 ▼'}
          </button>
          <div className="bg-bg rounded-lg p-3 max-h-48 overflow-auto font-mono text-xs text-fg/80 space-y-0.5">
            {logs.map((l, i) => (
              <div key={i}><span className="text-gray/50">[{l.t}]</span> {l.msg}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
