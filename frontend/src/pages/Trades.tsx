import { useEffect, useState, useCallback, useRef } from 'react'
import apiClient from '../api/client'
import type { Trade, TradesResponse } from '../api/types'
import { useWebSocket } from '../hooks/useWebSocket'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

type SortField = 'timestamp' | 'symbol' | 'side' | 'quantity' | 'price' | 'pnl'
type SortDir = 'asc' | 'desc'

function Trades() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [total, setTotal] = useState(0)
  const [symbolFilter, setSymbolFilter] = useState('')
  const [sortField, setSortField] = useState<SortField>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const allTradesRef = useRef<Trade[]>([])
  const { data: wsMessage } = useWebSocket(WS_URL)

  const limit = 20

  const fetchTrades = useCallback(async (p: number, sym: string) => {
    try {
      const params: Record<string, string | number> = { page: p, limit }
      if (sym) params.symbol = sym
      const resp = await apiClient.get<TradesResponse>('/trades', { params })
      setTrades(resp.data.trades)
      setTotalPages(resp.data.total_pages)
      setTotal(resp.data.total)
      setError(null)
    } catch {
      setError('Failed to fetch trades')
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchTrades(page, symbolFilter)
  }, [page, symbolFilter, fetchTrades])

  // WebSocket: receive new trades and full status updates
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const payload = wsMessage.payload as { trades?: Trade[] }
      if (payload.trades) {
        allTradesRef.current = payload.trades
        // Refresh current page from server for consistency
        fetchTrades(page, symbolFilter)
      }
    } else if (wsMessage?.type === 'trade') {
      // New trade arrived — refresh from server
      fetchTrades(page, symbolFilter)
    }
  }, [wsMessage, page, symbolFilter, fetchTrades])

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
      case 'timestamp':
        return dir * a.timestamp.localeCompare(b.timestamp)
      case 'symbol':
        return dir * a.symbol.localeCompare(b.symbol)
      case 'side':
        return dir * a.side.localeCompare(b.side)
      case 'quantity':
        return dir * (a.quantity - b.quantity)
      case 'price':
        return dir * (a.price - b.price)
      case 'pnl':
        return dir * ((a.pnl ?? 0) - (b.pnl ?? 0))
      default:
        return 0
    }
  })

  const exportCsv = () => {
    // Export all trades (fetch all pages for CSV)
    const exportAll = async () => {
      try {
        const params: Record<string, string | number> = { page: 1, limit: 10000 }
        if (symbolFilter) params.symbol = symbolFilter
        const resp = await apiClient.get<TradesResponse>('/trades', { params })
        const allTrades = resp.data.trades

        const header = 'Timestamp,Symbol,Side,Quantity,Price,PnL,Strategy'
        const rows = allTrades.map((t) =>
          `${t.timestamp},${t.symbol},${t.side},${t.quantity},${t.price},${t.pnl ?? ''},${t.strategy ?? ''}`
        )
        const csv = [header, ...rows].join('\n')
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`
        link.click()
        URL.revokeObjectURL(url)
      } catch {
        // Silently fail CSV export
      }
    }
    exportAll()
  }

  const SortIcon = ({ field }: { field: SortField }) => {
    if (field !== sortField) return <span className="text-gray-600 ml-1">&#8597;</span>
    return <span className="ml-1">{sortDir === 'asc' ? '&#9650;' : '&#9660;'}</span>
  }

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Trade History</h2>
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
        <h2 className="text-2xl font-bold mb-4">Trade History</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-bold">Trade History</h2>
          <span className="text-sm text-gray-400">{total} total</span>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Filter by symbol..."
            value={symbolFilter}
            onChange={(e) => {
              setSymbolFilter(e.target.value)
              setPage(1)
            }}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={exportCsv}
            className="bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200 transition-colors"
          >
            Export CSV
          </button>
        </div>
      </div>

      {sortedTrades.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
          <p className="text-gray-400 text-lg">No trades yet</p>
          <p className="text-gray-500 text-sm mt-1">
            {symbolFilter ? 'No trades found for this symbol.' : 'Trades will appear here when executed.'}
          </p>
        </div>
      ) : (
        <>
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('timestamp')}>
                    Time<SortIcon field="timestamp" />
                  </th>
                  <th className="text-left px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('symbol')}>
                    Symbol<SortIcon field="symbol" />
                  </th>
                  <th className="text-center px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('side')}>
                    Side<SortIcon field="side" />
                  </th>
                  <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('quantity')}>
                    Qty<SortIcon field="quantity" />
                  </th>
                  <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('price')}>
                    Price<SortIcon field="price" />
                  </th>
                  <th className="text-right px-4 py-3 font-medium cursor-pointer select-none" onClick={() => handleSort('pnl')}>
                    PnL<SortIcon field="pnl" />
                  </th>
                  <th className="text-left px-4 py-3 font-medium">Strategy</th>
                </tr>
              </thead>
              <tbody>
                {sortedTrades.map((trade, idx) => {
                  const isBuy = trade.side === 'BUY'
                  const hasPnl = trade.pnl != null
                  const isProfit = (trade.pnl ?? 0) >= 0
                  return (
                    <tr key={`${trade.timestamp}-${idx}`} className="border-b border-gray-700/50 hover:bg-gray-750">
                      <td className="px-4 py-3 text-gray-300">{new Date(trade.timestamp).toLocaleString()}</td>
                      <td className="px-4 py-3 font-medium text-white">{trade.symbol}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${
                          isBuy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {trade.side}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">{trade.quantity}</td>
                      <td className="px-4 py-3 text-right text-gray-300">${trade.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className={`px-4 py-3 text-right font-medium ${
                        hasPnl ? (isProfit ? 'text-green-400' : 'text-red-400') : 'text-gray-500'
                      }`}>
                        {hasPnl ? `${isProfit ? '+' : ''}${trade.pnl!.toFixed(2)}` : '--'}
                      </td>
                      <td className="px-4 py-3 text-gray-400">{trade.strategy ?? '--'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <span className="text-sm text-gray-400">
                Page {page} of {totalPages}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="px-3 py-1.5 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="px-3 py-1.5 rounded text-sm bg-gray-700 border border-gray-600 text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default Trades
