import { useState, useEffect, useRef } from 'react'

export function useWebSocket(url) {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const retryTimerRef = useRef(null)

  useEffect(() => {
    // url 为空时不建立连接（new WebSocket('') 会直接抛 SyntaxError）
    if (!url) {
      setConnected(false)
      return
    }

    let cancelled = false

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (cancelled) return
        setConnected(true)
      }
      ws.onmessage = (event) => {
        if (cancelled) return
        try {
          setData(JSON.parse(event.data))
        } catch (e) {}
      }
      ws.onclose = () => {
        // 组件已卸载：不再重连，否则会留下无人回收的连接
        if (cancelled) return
        setConnected(false)
        clearTimeout(retryTimerRef.current)
        retryTimerRef.current = setTimeout(connect, 3000)
      }
      ws.onerror = () => ws.close()
    }

    connect()

    return () => {
      cancelled = true
      clearTimeout(retryTimerRef.current)
      retryTimerRef.current = null
      const ws = wsRef.current
      wsRef.current = null
      if (ws) {
        // 先解绑回调再关闭，避免卸载后的迟到事件
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.onerror = null
        ws.close()
      }
    }
  }, [url])

  return { data, connected }
}
