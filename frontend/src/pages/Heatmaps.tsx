import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'
import type { HeatmapHourlyDow, HeatmapEngineSymbol, HeatmapMonthly } from '../api/types'

const DOW_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const PERIOD_OPTIONS = [
  { value: '', label: 'All Time' },
  { value: '7', label: '7 Days' },
  { value: '30', label: '30 Days' },
  { value: '90', label: '90 Days' },
]

function pnlColor(value: number, maxAbs: number): string {
  if (maxAbs === 0) return 'bg-gray-700'
  const intensity = Math.min(Math.abs(value) / maxAbs, 1)
  if (value > 0) {
    const alpha = Math.round(intensity * 80 + 20)
    return `rgba(34, 197, 94, ${alpha / 100})`
  } else if (value < 0) {
    const alpha = Math.round(intensity * 80 + 20)
    return `rgba(239, 68, 68, ${alpha / 100})`
  }
  return ''
}

function Tooltip({ text, x, y }: { text: string; x: number; y: number }) {
  return (
    <div
      className="fixed z-50 bg-gray-900 border border-gray-600 rounded-lg px-3 py-2 text-xs text-gray-200 shadow-lg pointer-events-none whitespace-pre-line"
      style={{ left: x + 12, top: y - 10 }}
    >
      {text}
    </div>
  )
}

function Heatmaps() {
  const [hourlyDow, setHourlyDow] = useState<HeatmapHourlyDow[]>([])
  const [engineSymbol, setEngineSymbol] = useState<HeatmapEngineSymbol[]>([])
  const [monthly, setMonthly] = useState<HeatmapMonthly[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [engineFilter, setEngineFilter] = useState('')
  const [period, setPeriod] = useState('')

  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null)

  const [yearFilter, setYearFilter] = useState(new Date().getFullYear())

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      const params: Record<string, string | number> = {}
      if (engineFilter) params.engine = engineFilter
      if (period) params.days = Number(period)

      const [h, e, m] = await Promise.all([
        apiClient.get<{ data: HeatmapHourlyDow[] }>('/analytics/heatmap', { params: { ...params, type: 'hourly_dow' } }),
        apiClient.get<{ data: HeatmapEngineSymbol[] }>('/analytics/heatmap', { params: { ...params, type: 'engine_symbol' } }),
        apiClient.get<{ data: HeatmapMonthly[] }>('/analytics/heatmap', { params: { ...params, type: 'monthly' } }),
      ])
      setHourlyDow(h.data.data)
      setEngineSymbol(e.data.data)
      setMonthly(m.data.data)
      setError(null)
    } catch {
      setError('Failed to fetch heatmap data')
    } finally {
      setLoading(false)
    }
  }, [engineFilter, period])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Hourly x DOW grid
  const hourlyMaxAbs = Math.max(...hourlyDow.map((d) => Math.abs(d.pnl)), 0.01)

  // Engine x Symbol grid
  const esEngines = [...new Set(engineSymbol.map((d) => d.engine))].sort()
  const esSymbols = [...new Set(engineSymbol.map((d) => d.symbol))].sort()
  const esMap = new Map(engineSymbol.map((d) => [`${d.engine}|${d.symbol}`, d]))
  const esMaxAbs = Math.max(...engineSymbol.map((d) => Math.abs(d.pnl)), 0.01)

  // Monthly calendar filtered by year
  const filteredMonthly = monthly.filter((d) => d.year === yearFilter)
  const monthlyMaxAbs = Math.max(...monthly.map((d) => Math.abs(d.pnl)), 0.01)
  const availableYears = [...new Set(monthly.map((d) => d.year))].sort()
  if (availableYears.length === 0) availableYears.push(new Date().getFullYear())

  const showTooltip = (e: React.MouseEvent, text: string) => {
    setTooltip({ text, x: e.clientX, y: e.clientY })
  }
  const hideTooltip = () => setTooltip(null)

  if (loading && hourlyDow.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Performance Heatmaps</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4 w-1/3" />
          <div className="h-64 bg-gray-700 rounded" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Performance Heatmaps</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">{error}</div>
      </div>
    )
  }

  return (
    <div>
      {tooltip && <Tooltip text={tooltip.text} x={tooltip.x} y={tooltip.y} />}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <h2 className="text-2xl font-bold">Performance Heatmaps</h2>
        <div className="flex items-center gap-3">
          <select
            value={engineFilter}
            onChange={(e) => setEngineFilter(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          >
            <option value="">All Engines</option>
            <option value="funding_rate_arb">Funding Arb</option>
            <option value="grid_trading">Grid Trading</option>
            <option value="cross_exchange_arb">Cross-Exchange Arb</option>
            <option value="stat_arb">Stat Arb</option>
          </select>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          >
            {PERIOD_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Hourly x Day-of-Week Heatmap */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h3 className="text-lg font-bold mb-4">PnL by Hour &amp; Day of Week</h3>
        <div className="overflow-x-auto">
          <div className="inline-block">
            {/* Hour headers */}
            <div className="flex">
              <div className="w-12 shrink-0" />
              {Array.from({ length: 24 }, (_, h) => (
                <div key={h} className="w-8 h-6 flex items-center justify-center text-xs text-gray-500">
                  {h}
                </div>
              ))}
            </div>
            {/* Grid rows */}
            {DOW_LABELS.map((label, dow) => (
              <div key={dow} className="flex">
                <div className="w-12 shrink-0 flex items-center text-xs text-gray-400 font-medium">
                  {label}
                </div>
                {Array.from({ length: 24 }, (_, hour) => {
                  const cell = hourlyDow.find((d) => d.dow === dow && d.hour === hour)
                  const pnl = cell?.pnl ?? 0
                  const count = cell?.trade_count ?? 0
                  const winRate = cell?.win_rate ?? 0
                  const bg = count > 0 ? pnlColor(pnl, hourlyMaxAbs) : ''
                  return (
                    <div
                      key={hour}
                      className="w-8 h-8 border border-gray-700/50 rounded-sm cursor-default flex items-center justify-center"
                      style={{ backgroundColor: bg || undefined }}
                      onMouseEnter={(e) =>
                        showTooltip(
                          e,
                          `${label} ${hour}:00\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                        )
                      }
                      onMouseMove={(e) =>
                        showTooltip(
                          e,
                          `${label} ${hour}:00\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                        )
                      }
                      onMouseLeave={hideTooltip}
                    >
                      {count > 0 && (
                        <span className="text-[10px] text-white/70 font-mono">{count}</span>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
            {/* Legend */}
            <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(239, 68, 68, 0.8)' }} />
                Loss
              </span>
              <span className="flex items-center gap-1">
                <div className="w-4 h-4 rounded bg-gray-700" />
                No trades
              </span>
              <span className="flex items-center gap-1">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(34, 197, 94, 0.8)' }} />
                Profit
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Engine x Symbol Matrix */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h3 className="text-lg font-bold mb-4">PnL by Engine &amp; Symbol</h3>
        {esEngines.length === 0 ? (
          <p className="text-gray-500 text-sm">No trade data available.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-sm">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-gray-400 font-medium" />
                  {esSymbols.map((s) => (
                    <th key={s} className="px-3 py-2 text-center text-gray-400 font-medium text-xs">{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {esEngines.map((eng) => (
                  <tr key={eng}>
                    <td className="px-3 py-2 text-gray-300 text-xs font-medium whitespace-nowrap">{eng}</td>
                    {esSymbols.map((sym) => {
                      const cell = esMap.get(`${eng}|${sym}`)
                      const pnl = cell?.pnl ?? 0
                      const count = cell?.trade_count ?? 0
                      const winRate = cell?.win_rate ?? 0
                      const bg = count > 0 ? pnlColor(pnl, esMaxAbs) : ''
                      return (
                        <td
                          key={sym}
                          className="px-3 py-2 text-center cursor-default"
                          style={{ backgroundColor: bg || undefined }}
                          onMouseEnter={(e) =>
                            showTooltip(
                              e,
                              `${eng} / ${sym}\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                            )
                          }
                          onMouseMove={(e) =>
                            showTooltip(
                              e,
                              `${eng} / ${sym}\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                            )
                          }
                          onMouseLeave={hideTooltip}
                        >
                          {count > 0 ? (
                            <span className={`text-xs font-mono ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-xs text-gray-600">-</span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Monthly PnL Calendar */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">Monthly PnL Calendar</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setYearFilter((y) => y - 1)}
              className="px-2 py-1 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 hover:bg-gray-600 transition-colors"
            >
              &lt;
            </button>
            <span className="text-sm text-gray-300 font-medium w-12 text-center">{yearFilter}</span>
            <button
              onClick={() => setYearFilter((y) => y + 1)}
              className="px-2 py-1 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 hover:bg-gray-600 transition-colors"
            >
              &gt;
            </button>
          </div>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
          {Array.from({ length: 12 }, (_, i) => {
            const monthData = filteredMonthly.find((d) => d.month === i + 1)
            const pnl = monthData?.pnl ?? 0
            const count = monthData?.trade_count ?? 0
            const winRate = monthData?.win_rate ?? 0
            const bg = count > 0 ? pnlColor(pnl, monthlyMaxAbs) : ''
            return (
              <div
                key={i}
                className="rounded-lg border border-gray-700/50 p-3 cursor-default"
                style={{ backgroundColor: bg || undefined }}
                onMouseEnter={(e) =>
                  showTooltip(
                    e,
                    `${MONTH_LABELS[i]} ${yearFilter}\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                  )
                }
                onMouseMove={(e) =>
                  showTooltip(
                    e,
                    `${MONTH_LABELS[i]} ${yearFilter}\nPnL: ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}\nTrades: ${count}\nWin Rate: ${(winRate * 100).toFixed(0)}%`
                  )
                }
                onMouseLeave={hideTooltip}
              >
                <div className="text-xs text-gray-400 mb-1">{MONTH_LABELS[i]}</div>
                {count > 0 ? (
                  <>
                    <div className={`text-sm font-mono font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                    </div>
                    <div className="text-[10px] text-gray-500 mt-0.5">{count} trades</div>
                  </>
                ) : (
                  <div className="text-xs text-gray-600">No data</div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default Heatmaps
