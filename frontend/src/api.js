// 给后端请求统一带上当前界面语言。
//
// 为什么需要：后端有一部分提示是运行时拼出来的（检测结果、安装日志、启停消息），
// 前端拿不到"未渲染的素材"，只能原样显示，所以由后端按 ?lang= 决定文案语言
// （见 backend/app/services/i18n.py）。在这里统一注入，避免每个 fetch 各写一遍。
//
// 语言来源与 i18n/I18nContext.jsx 共用同一个 localStorage key。

const STORAGE_KEY = 'readyllm_lang'

function currentLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'en' || saved === 'zh') return saved
  } catch { /* ignore */ }
  return 'en'
}

export function withLang(url) {
  if (typeof url !== 'string') return url
  if (!url.includes('/api/')) return url
  if (/[?&]lang=/.test(url)) return url
  return `${url}${url.includes('?') ? '&' : '?'}lang=${currentLang()}`
}

// 副作用引入一次即可对所有 fetch 生效（main.jsx 里 import './api'）
const nativeFetch = window.fetch.bind(window)

window.fetch = (input, init) => {
  if (typeof input === 'string') {
    return nativeFetch(withLang(input), init)
  }
  if (input instanceof Request) {
    return nativeFetch(new Request(withLang(input.url), input), init)
  }
  return nativeFetch(input, init)
}