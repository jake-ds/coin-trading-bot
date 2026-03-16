import { useEffect, useState } from 'react'
import apiClient from '../api/client'
import type { Position, PositionsResponse } from '../api/types'
import { useWebSocket } from '../hooks/useWebSocket'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`

interface FundingItem {
  time: number
  symbol: string
  amount: number
}

function formatDuration(openedAt: string | undefined): string {
  if (!openedAt) return '--'
  const opened = new Date(openedAt).getTime()
  if (isNaN(opened)) return '--'
  const now = Date.now()
  const diffMs = now - opened
  const hours = Math.floor(diffMs / 3600000)
  const mins = Math.floor((diffMs % 3600000) / 60000)
  if (hours > 24) {
    const days = Math.floor(hours / 24)
    return `${days}d ${hours % 24}h`
  }
  return `${hours}h ${mins}m`
}

function Positions() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fundingIncome, setFundingIncome] = useState<FundingItem[]>([])
  const [fundingTotal, setFundingTotal] = useState(0)
  const { data: wsMessage } = useWebSocket(WS_URL)

  // Apply WebSocket updates
  useEffect(() => {
    if (wsMessage?.type === 'status_update') {
      const payload = wsMessage.payload as { open_positions?: Position[] }
      if (payload.open_positions) {
        setPositions(payload.open_positions)
        setLoading(false)
      }
    } else if (wsMessage?.type === 'position_change') {
      const payload = wsMessage.payload as { positions?: Position[] }
      if (payload.positions) {
        setPositions(payload.positions)
      }
    }
  }, [wsMessage])

  // Initial fetch
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const resp = await apiClient.get<PositionsResponse>('/positions')
        setPositions(resp.data.positions)
        setError(null)
      } catch {
        setError('Failed to fetch positions')
      } finally {
        setLoading(false)
      }
    }
    fetchPositions()
  }, [])

  // Fetch funding income
  useEffect(() => {
    const fetchFunding = async () => {
      try {
        const resp = await apiClient.get<{ income: FundingItem[]; total: number }>('/funding-income')
        setFundingIncome(resp.data.income)
        setFundingTotal(resp.data.total)
      } catch { /* ignore */ }
    }
    fetchFunding()
    const interval = setInterval(fetchFunding, 60000) // refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Open Positions</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4 w-1/3" />
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-gray-700 rounded" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Open Positions</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      </div>
    )
  }

  const totalPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0)
  const isTotalProfit = totalPnl >= 0
  const isFundingProfit = fundingTotal >= 0

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold">Open Positions</h2>
        <span className="text-sm text-gray-400">{positions.length} position{positions.length !== 1 ? 's' : ''}</span>
      </div>

      {positions.length > 0 && (
        <div className={`rounded-lg p-4 mb-4 border ${isTotalProfit ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
          <div className="text-sm text-gray-400 mb-1">Total Unrealized PnL</div>
          <div className={`text-3xl font-bold ${isTotalProfit ? 'text-green-400' : 'text-red-400'}`}>
            {isTotalProfit ? '+' : ''}{totalPnl.toFixed(2)} USDT
          </div>
        </div>
      )}

      {positions.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
          <p className="text-gray-400 text-lg">No open positions</p>
          <p className="text-gray-500 text-sm mt-1">Positions will appear here when trades are executed.</p>
        </div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400">
                <th className="text-left px-4 py-3 font-medium">Symbol</th>
                <th className="text-left px-4 py-3 font-medium">Engine</th>
                <th className="text-right px-4 py-3 font-medium">Qty</th>
                <th className="text-right px-4 py-3 font-medium">Entry Price</th>
                <th className="text-right px-4 py-3 font-medium">Current Price</th>
                <th className="text-right px-4 py-3 font-medium">Unrealized PnL</th>
                <th className="text-right px-4 py-3 font-medium">Stop Loss</th>
                <th className="text-right px-4 py-3 font-medium">Take Profit</th>
                <th className="text-right px-4 py-3 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, idx) => {
                const pnlPct = pos.entry_price > 0
                  ? ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                  : 0
                const isProfit = pos.unrealized_pnl >= 0
                return (
                  <tr key={`${pos.symbol}-${idx}`} className="border-b border-gray-700/50 hover:bg-gray-750">
                    <td className="px-4 py-3 font-medium text-white">{pos.symbol}</td>
                    <td className="px-4 py-3 text-gray-300 text-sm">{pos.strategy?.startsWith('engine:') ? pos.strategy.slice(7) : pos.strategy || '--'}</td>
                    <td className="px-4 py-3 text-right text-gray-300">{pos.quantity}</td>
                    <td className="px-4 py-3 text-right text-gray-300">${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                    <td className="px-4 py-3 text-right text-gray-300">${pos.current_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                    <td className={`px-4 py-3 text-right font-medium rounded ${isProfit ? 'text-green-400 bg-green-500/10' : 'text-red-400 bg-red-500/10'}`}>
                      <div>{isProfit ? '+' : ''}{pos.unrealized_pnl.toFixed(2)}</div>
                      <div className="text-xs opacity-75">{isProfit ? '+' : ''}{pnlPct.toFixed(2)}%</div>
                    </td>
                    <td className="px-4 py-3 text-right text-gray-400">{pos.stop_loss > 0 ? `$${pos.stop_loss.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : '--'}</td>
                    <td className="px-4 py-3 text-right text-gray-400">{pos.take_profit > 0 ? `$${pos.take_profit.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : '--'}</td>
                    <td className="px-4 py-3 text-right text-gray-400">{formatDuration(pos.opened_at)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Funding Fee Income */}
      <div className="mt-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Funding Fee Income</h2>
          <div className={`text-lg font-bold ${isFundingProfit ? 'text-green-400' : 'text-red-400'}`}>
            Total: {isFundingProfit ? '+' : ''}{fundingTotal.toFixed(4)} USDT
          </div>
        </div>

        {fundingIncome.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
            <p className="text-gray-400">No funding fee income yet</p>
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left px-4 py-3 font-medium">Time</th>
                  <th className="text-left px-4 py-3 font-medium">Symbol</th>
                  <th className="text-right px-4 py-3 font-medium">Amount</th>
                </tr>
              </thead>
              <tbody>
                {[...fundingIncome].reverse().map((item, idx) => {
                  const isPositive = item.amount >= 0
                  const dt = new Date(item.time)
                  const timeStr = dt.toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
                  return (
                    <tr key={`funding-${idx}`} className="border-b border-gray-700/50">
                      <td className="px-4 py-3 text-gray-300">{timeStr}</td>
                      <td className="px-4 py-3 font-medium text-white">{item.symbol}</td>
                      <td className={`px-4 py-3 text-right font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {isPositive ? '+' : ''}{item.amount.toFixed(4)} USDT
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Positions
