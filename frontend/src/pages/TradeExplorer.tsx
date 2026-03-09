import { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import apiClient from '../api/client'
import type { TradeDetail, TradeDetailResponse } from '../api/types'

type SortField = 'exit_time' | 'symbol' | 'engine_name' | 'net_pnl' | 'hold_time_seconds'
type SortDir = 'asc' | 'desc'

function formatHoldTime(seconds: number): string {
  if (seconds <= 0) return '0m'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function TradeExplorer() {
  const [trades, setTrades] = useState<TradeDetail[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const limit = 20

  // Filters
  const [engineFilter, setEngineFilter] = useState('')
  const [symbolFilter, setSymbolFilter] = useState('')
  const [winFilter, setWinFilter] = useState<'' | 'win' | 'loss'>('')

  // Sort
  const [sortField, setSortField] = useState<SortField>('exit_time')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  // Expanded row
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)

  const fetchTrades = useCallback(async () => {
    try {
      setLoading(true)
      const params: Record<string, string | number | boolean> = { limit, offset }
      if (engineFilter) params.engine = engineFilter
      if (symbolFilter) params.symbol = symbolFilter
      if (winFilter === 'win') params.win_only = true
      if (winFilter === 'loss') params.win_only = false
      const resp = await apiClient.get<TradeDetailResponse>('/trades/detail', { params })
      setTrades(resp.data.trades)
      setTotal(resp.data.total)
      setError(null)
    } catch {
      setError('Failed to fetch trade details')
    } finally {
      setLoading(false)
    }
  }, [offset, engineFilter, symbolFilter, winFilter])

  useEffect(() => {
    fetchTrades()
  }, [fetchTrades])

  // Reset pagination when filters change
  useEffect(() => {
    setOffset(0)
    setExpandedIdx(null)
  }, [engineFilter, symbolFilter, winFilter])

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const sortedTrades = [...trades].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1
    switch (sortField) {
      case 'exit_time':
        return dir * a.exit_time.localeCompare(b.exit_time)
      case 'symbol':
        return dir * a.symbol.localeCompare(b.symbol)
      case 'engine_name':
        return dir * a.engine_name.localeCompare(b.engine_name)
      case 'net_pnl':
        return dir * (a.net_pnl - b.net_pnl)
      case 'hold_time_seconds':
        return dir * (a.hold_time_seconds - b.hold_time_seconds)
      default:
        return 0
    }
  })

  const exportCsv = async () => {
    try {
      const params: Record<string, string> = {}
      if (engineFilter) params.engine = engineFilter
      if (symbolFilter) params.symbol = symbolFilter
      const resp = await apiClient.get('/trades/export', { params, responseType: 'blob' })
      const blob = new Blob([resp.data as BlobPart], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`
      link.click()
      URL.revokeObjectURL(url)
    } catch {
      // Silently fail
    }
  }

  const totalPages = Math.max(1, Math.ceil(total / limit))
  const currentPage = Math.floor(offset / limit) + 1

  const SortIcon = ({ field }: { field: SortField }) => {
    if (field !== sortField) return <span className="text-gray-600 ml-1">&#8597;</span>
    return <span className="ml-1">{sortDir === 'asc' ? '\u25B2' : '\u25BC'}</span>
  }

  if (loading && trades.length === 0) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Trade Explorer</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4 w-1/3" />
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-gray-700 rounded" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Trade Explorer</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">{error}</div>
      </div>
    )
  }

  const expandedTrade = expandedIdx !== null ? sortedTrades[expandedIdx] : null

  // PnL breakdown for waterfall chart
  const waterfallData = expandedTrade
    ? [
        { name: 'Gross', value: expandedTrade.pnl, fill: expandedTrade.pnl >= 0 ? '#22c55e' : '#ef4444' },
        { name: 'Cost', value: -Math.abs(expandedTrade.cost), fill: '#f59e0b' },
        { name: 'Net', value: expandedTrade.net_pnl, fill: expandedTrade.net_pnl >= 0 ? '#22c55e' : '#ef4444' },
      ]
    : []

  // Same-symbol history
  const sameSymbolHistory = expandedTrade
    ? trades.filter((t) => t.symbol === expandedTrade.symbol && t !== expandedTrade).slice(0, 10)
    : []

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold">Trade Explorer</h2>
          <span className="text-sm text-gray-400">{total} trades</span>
        </div>
        <button
          onClick={exportCsv}
          className="bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 transition-colors"
        >
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
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
        <input
          type="text"
          placeholder="Symbol filter..."
          value={symbolFilter}
          onChange={(e) => setSymbolFilter(e.target.value)}
          className="bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <select
          value={winFilter}
          onChange={(e) => setWinFilter(e.target.value as '' | 'win' | 'loss')}
          className="bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
        >
          <option value="">All Trades</option>
          <option value="win">Wins Only</option>
          <option value="loss">Losses Only</option>
        </select>
      </div>

      {/* Trade Table */}
      {sortedTrades.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
          <p className="text-gray-400 text-lg">No trades found</p>
          <p className="text-gray-500 text-sm mt-1">Adjust filters or wait for trades to execute.</p>
        </div>
      ) : (
        <>
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('exit_time')}>
                    Time<SortIcon field="exit_time" />
                  </th>
                  <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('engine_name')}>
                    Engine<SortIcon field="engine_name" />
                  </th>
                  <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('symbol')}>
                    Symbol<SortIcon field="symbol" />
                  </th>
                  <th className="text-center px-4 py-3 font-medium">Side</th>
                  <th className="text-right px-4 py-3 font-medium">Entry</th>
                  <th className="text-right px-4 py-3 font-medium">Exit</th>
                  <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('net_pnl')}>
                    Net PnL<SortIcon field="net_pnl" />
                  </th>
                  <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('hold_time_seconds')}>
                    Hold<SortIcon field="hold_time_seconds" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedTrades.map((trade, idx) => {
                  const isProfit = trade.net_pnl >= 0
                  const isExpanded = expandedIdx === idx
                  return (
                    <tr
                      key={`${trade.exit_time}-${idx}`}
                      className={`border-b border-gray-700/50 cursor-pointer transition-colors ${
                        isExpanded ? 'bg-gray-700/50' : 'hover:bg-gray-750'
                      }`}
                      onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                    >
                      <td className="px-4 py-3 text-gray-300">{new Date(trade.exit_time).toLocaleString()}</td>
                      <td className="px-4 py-3 text-gray-400">{trade.engine_name}</td>
                      <td className="px-4 py-3 font-medium text-white">{trade.symbol}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                          trade.side.toLowerCase().includes('buy') || trade.side.toLowerCase().includes('long')
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-red-500/20 text-red-400'
                        }`}>
                          {trade.side}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        ${trade.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        ${trade.exit_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                      </td>
                      <td className={`px-4 py-3 text-right font-medium ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                        {isProfit ? '+' : ''}{trade.net_pnl.toFixed(4)}
                      </td>
                      <td className="px-4 py-3 text-right text-gray-400">{formatHoldTime(trade.hold_time_seconds)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <span className="text-sm text-gray-400">Page {currentPage} of {totalPages}</span>
              <div className="flex gap-2">
                <button
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                  disabled={offset === 0}
                  className="px-3 py-1.5 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                >
                  Previous
                </button>
                <button
                  onClick={() => setOffset(offset + limit)}
                  disabled={offset + limit >= total}
                  className="px-3 py-1.5 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Expanded Detail Panel */}
      {expandedTrade && (
        <div className="mt-4 bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold mb-4">
            {expandedTrade.symbol} — {expandedTrade.engine_name}
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            <div>
              <div className="text-xs text-gray-500">Entry Price</div>
              <div className="text-white font-medium">
                ${expandedTrade.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Exit Price</div>
              <div className="text-white font-medium">
                ${expandedTrade.exit_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Quantity</div>
              <div className="text-white font-medium">{expandedTrade.quantity}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Hold Time</div>
              <div className="text-white font-medium">{formatHoldTime(expandedTrade.hold_time_seconds)}</div>
            </div>
          </div>

          {/* PnL Waterfall Chart */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-400 mb-2">PnL Breakdown</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={waterfallData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                    labelStyle={{ color: '#e5e7eb' }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {waterfallData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-6 mt-2 text-sm">
              <span className="text-gray-400">
                Gross: <span className={expandedTrade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>{expandedTrade.pnl >= 0 ? '+' : ''}{expandedTrade.pnl.toFixed(4)}</span>
              </span>
              <span className="text-gray-400">
                Cost: <span className="text-amber-400">-{Math.abs(expandedTrade.cost).toFixed(4)}</span>
              </span>
              <span className="text-gray-400">
                Net: <span className={expandedTrade.net_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>{expandedTrade.net_pnl >= 0 ? '+' : ''}{expandedTrade.net_pnl.toFixed(4)}</span>
              </span>
            </div>
          </div>

          {/* Same Symbol History */}
          {sameSymbolHistory.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-2">
                Recent {expandedTrade.symbol} Trades
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-700">
                      <th className="text-left px-3 py-2">Time</th>
                      <th className="text-left px-3 py-2">Engine</th>
                      <th className="text-center px-3 py-2">Side</th>
                      <th className="text-right px-3 py-2">Net PnL</th>
                      <th className="text-right px-3 py-2">Hold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sameSymbolHistory.map((t, i) => (
                      <tr key={i} className="border-b border-gray-700/30">
                        <td className="px-3 py-1.5 text-gray-400">{new Date(t.exit_time).toLocaleString()}</td>
                        <td className="px-3 py-1.5 text-gray-400">{t.engine_name}</td>
                        <td className="px-3 py-1.5 text-center text-gray-400">{t.side}</td>
                        <td className={`px-3 py-1.5 text-right ${t.net_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {t.net_pnl >= 0 ? '+' : ''}{t.net_pnl.toFixed(4)}
                        </td>
                        <td className="px-3 py-1.5 text-right text-gray-400">{formatHoldTime(t.hold_time_seconds)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default TradeExplorer
