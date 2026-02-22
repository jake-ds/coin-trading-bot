import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'
import type { EngineInfo, EnginesResponse } from '../api/types'

const statusColors: Record<string, string> = {
  running: 'bg-green-500',
  paused: 'bg-yellow-500',
  stopped: 'bg-gray-500',
  error: 'bg-red-500',
}

function Engines() {
  const [engines, setEngines] = useState<Record<string, EngineInfo>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)

  const fetchEngines = useCallback(async () => {
    try {
      const res = await apiClient.get<EnginesResponse>('/engines')
      setEngines(res.data)
      setError(null)
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } }
        if (axiosErr.response?.status === 503) {
          setError('Engine mode is not enabled. Set engine_mode=True to use multi-engine trading.')
        } else {
          setError('Failed to fetch engines')
        }
      } else {
        setError('Failed to fetch engines')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchEngines()
    const interval = setInterval(fetchEngines, 5000)
    return () => clearInterval(interval)
  }, [fetchEngines])

  const handleAction = async (name: string, action: string) => {
    setActionLoading(`${name}-${action}`)
    try {
      await apiClient.post(`/engines/${name}/${action}`)
      await fetchEngines()
    } catch {
      // Error handled by next poll
    } finally {
      setActionLoading(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading engines...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <p className="text-yellow-400">{error}</p>
      </div>
    )
  }

  const engineList = Object.values(engines)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Trading Engines</h2>
        <span className="text-sm text-gray-400">
          {engineList.filter((e) => e.status === 'running').length} / {engineList.length} running
        </span>
      </div>

      {engineList.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 text-center text-gray-400">
          No engines registered
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {engineList.map((engine) => (
            <div
              key={engine.name}
              className="bg-gray-800 rounded-lg p-5 border border-gray-700"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${statusColors[engine.status] || 'bg-gray-500'}`}
                  />
                  <h3 className="font-semibold text-lg">{engine.name}</h3>
                </div>
                <span className="text-xs px-2 py-0.5 rounded bg-gray-700 text-gray-300 uppercase">
                  {engine.status}
                </span>
              </div>

              <p className="text-sm text-gray-400 mb-4">{engine.description}</p>

              <div className="grid grid-cols-2 gap-3 text-sm mb-4">
                <div>
                  <span className="text-gray-500">Capital</span>
                  <p className="font-mono">${engine.allocated_capital.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-500">PnL</span>
                  <p
                    className={`font-mono ${engine.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    ${engine.total_pnl >= 0 ? '+' : ''}
                    {engine.total_pnl.toFixed(2)}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Positions</span>
                  <p className="font-mono">
                    {engine.position_count} / {engine.max_positions}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Cycles</span>
                  <p className="font-mono">{engine.cycle_count}</p>
                </div>
              </div>

              {engine.error && (
                <div className="text-xs text-red-400 bg-red-900/20 rounded p-2 mb-3">
                  {engine.error}
                </div>
              )}

              <div className="flex gap-2">
                {engine.status === 'stopped' && (
                  <button
                    className="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded text-sm disabled:opacity-50"
                    onClick={() => handleAction(engine.name, 'start')}
                    disabled={actionLoading === `${engine.name}-start`}
                  >
                    Start
                  </button>
                )}
                {engine.status === 'running' && (
                  <>
                    <button
                      className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-500 rounded text-sm disabled:opacity-50"
                      onClick={() => handleAction(engine.name, 'pause')}
                      disabled={actionLoading === `${engine.name}-pause`}
                    >
                      Pause
                    </button>
                    <button
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-sm disabled:opacity-50"
                      onClick={() => handleAction(engine.name, 'stop')}
                      disabled={actionLoading === `${engine.name}-stop`}
                    >
                      Stop
                    </button>
                  </>
                )}
                {engine.status === 'paused' && (
                  <>
                    <button
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm disabled:opacity-50"
                      onClick={() => handleAction(engine.name, 'resume')}
                      disabled={actionLoading === `${engine.name}-resume`}
                    >
                      Resume
                    </button>
                    <button
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded text-sm disabled:opacity-50"
                      onClick={() => handleAction(engine.name, 'stop')}
                      disabled={actionLoading === `${engine.name}-stop`}
                    >
                      Stop
                    </button>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default Engines
