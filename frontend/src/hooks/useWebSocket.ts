import { useEffect, useRef, useState, useCallback } from 'react'

export interface WebSocketMessage {
  type: 'status_update' | 'trade' | 'position_change' | 'alert'
  payload: Record<string, unknown>
}

interface UseWebSocketReturn {
  connected: boolean
  data: WebSocketMessage | null
  lastUpdate: number | null
}

const MAX_RECONNECT_DELAY = 30000
const INITIAL_RECONNECT_DELAY = 1000

export function useWebSocket(url: string): UseWebSocketReturn {
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState<WebSocketMessage | null>(null)
  const [lastUpdate, setLastUpdate] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setConnected(true)
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const message = JSON.parse(event.data) as WebSocketMessage
          setData(message)
          setLastUpdate(Date.now())
        } catch {
          // Ignore malformed messages
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        wsRef.current = null
        // Schedule reconnect with exponential backoff
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(
          delay * 2,
          MAX_RECONNECT_DELAY
        )
        reconnectTimerRef.current = setTimeout(connect, delay)
      }

      ws.onerror = () => {
        // onclose will fire after onerror, which handles reconnect
        ws.close()
      }
    } catch {
      // Schedule reconnect on connection failure
      if (mountedRef.current) {
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(
          delay * 2,
          MAX_RECONNECT_DELAY
        )
        reconnectTimerRef.current = setTimeout(connect, delay)
      }
    }
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { connected, data, lastUpdate }
}
