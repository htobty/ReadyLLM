import { createContext, useContext, useState, useEffect } from 'react'
import { translations } from './translations'

const I18nContext = createContext(null)
const STORAGE_KEY = 'readyllm_lang'

export function I18nProvider({ children }) {
  const [lang, setLang] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved === 'en' || saved === 'zh') return saved
    } catch { /* ignore */ }
    return 'en' // 默认英文
  })

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, lang) } catch { /* ignore */ }
  }, [lang])

  // t(key, vars)：取当前语言文案，缺失回退英文，再缺失回退 key 本身；
  // 支持 {var} 插值，如 t('deploy.modelCount', { n: 5 })
  const t = (key, vars) => {
    const dict = translations[lang] || translations.en
    let s = dict[key] ?? translations.en[key] ?? key
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        s = s.split(`{${k}}`).join(String(v))
      }
    }
    return s
  }

  const toggle = () => setLang(l => (l === 'en' ? 'zh' : 'en'))

  return (
    <I18nContext.Provider value={{ lang, setLang, toggle, t }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n() {
  return useContext(I18nContext)
}

// 语言切换器（放侧边栏底部）
export function LangSwitch() {
  const { lang, toggle } = useI18n()
  return (
    <button
      onClick={toggle}
      className="mt-3 w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm border border-white/10 text-fg/70 hover:text-fg hover:bg-white/[0.04] transition"
      title={lang === 'en' ? '切换到中文' : 'Switch to English'}
    >
      <span>🌐</span>
      <span>{lang === 'en' ? '中文' : 'English'}</span>
    </button>
  )
}
