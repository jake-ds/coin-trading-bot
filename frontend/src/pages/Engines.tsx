import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import apiClient from '../api/client'
import type { EngineInfo, EnginesResponse } from '../api/types'

const statusColors: Record<string, string> = {
  running: 'bg-green-500',
  paused: 'bg-yellow-500',
  stopped: 'bg-gray-500',
  error: 'bg-red-500',
}

/** Chevron that rotates when expanded */
function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-90' : ''}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
}

/** Collapsible description section for an engine card */
function EngineDescription({
  engine,
  params,
}: {
  engine: EngineInfo
  params: Record<string, unknown> | null
}) {
  const hasDescription = engine.role_ko || engine.description_ko

  if (!hasDescription) return null

  return (
    <div className="space-y-3 text-sm">
      {/* Role name + English explanation */}
      <div>
        {engine.role_ko && (
          <p className="font-semibold text-gray-100">{engine.role_ko}</p>
        )}
        {engine.role_en && (
          <p className="text-gray-400 text-xs">{engine.role_en}</p>
        )}
      </div>

      {/* How it works */}
      {engine.description_ko && (
        <p className="text-gray-400 text-xs leading-relaxed">
          {engine.description_ko}
        </p>
      )}

      {/* Tracked Symbols */}
      {engine.symbols && engine.symbols.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1.5">Tracked Symbols</p>
          <div className="flex flex-wrap gap-1.5">
            {engine.symbols.map((sym, i) => {
              const label = Array.isArray(sym) ? sym.join(' / ') : sym
              return (
                <span
                  key={i}
                  className="text-xs px-2 py-0.5 rounded-full bg-gray-700 text-gray-300"
                >
                  {label}
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* Key Parameters */}
      {params && Object.keys(params).length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1.5">Key Parameters</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {Object.entries(params).map(([key, value]) => (
              <div key={key} className="flex justify-between text-xs">
                <span className="text-gray-400 truncate">{key}</span>
                <span className="font-mono text-gray-200 ml-2">
                  {formatParamValue(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cost info */}
      {engine.key_params && (
        <div className="text-xs text-gray-500">
          <span>Key: </span>
          <span className="text-gray-400">{engine.key_params}</span>
        </div>
      )}
    </div>
  )
}

function formatParamValue(value: unknown): string {
  if (Array.isArray(value)) {
    if (value.length <= 3) return value.join(', ')
    return `${value.slice(0, 2).join(', ')}... (${value.length})`
  }
  if (typeof value === 'number') {
    return value % 1 === 0 ? String(value) : value.toFixed(4)
  }
  return String(value)
}

function Engines() {
  const [engines, setEngines] = useState<Record<string, EngineInfo>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [engineParams, setEngineParams] = useState<
    Record<string, Record<string, unknown>>
  >({})

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

  // Fetch params when an engine is expanded
  const toggleExpand = useCallback(
    async (name: string) => {
      setExpanded((prev) => {
        const next = { ...prev, [name]: !prev[name] }
        // Fetch params on first expand
        if (next[name] && !engineParams[name]) {
          apiClient
            .get(`/engines/${name}/params`)
            .then((res) => {
              setEngineParams((prev) => ({
                ...prev,
                [name]: res.data.params || {},
              }))
            })
            .catch(() => {
              // Params not available — OK
            })
        }
        return next
      })
    },
    [engineParams],
  )

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

              <p className="text-sm text-gray-400 mb-3">{engine.description}</p>

              {/* Collapsible description section */}
              {(engine.role_ko || engine.description_ko) && (
                <div className="mb-4">
                  <button
                    onClick={() => toggleExpand(engine.name)}
                    className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-200 transition-colors mb-2"
                  >
                    <ChevronIcon expanded={!!expanded[engine.name]} />
                    <span>Details</span>
                  </button>
                  {expanded[engine.name] && (
                    <div className="bg-gray-900/50 rounded p-3 border border-gray-700/50">
                      <EngineDescription
                        engine={engine}
                        params={engineParams[engine.name] || null}
                      />
                    </div>
                  )}
                </div>
              )}

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

              <div className="flex gap-2 items-center">
                <Link
                  to={`/engines/${engine.name}`}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-200"
                >
                  View Details
                </Link>
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
