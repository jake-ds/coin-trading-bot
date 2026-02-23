import { useState, useEffect, useCallback } from 'react'
import type { ScannerResponse, ScannerOpportunity, OpportunityTypeName } from '../api/types'

const API_BASE = '/api'

const TYPE_LABELS: Record<OpportunityTypeName, { label: string; emoji: string; color: string }> = {
  funding_rate: { label: 'Funding Rate', emoji: '\u{1F4B0}', color: 'blue' },
  volatility: { label: 'Volatility', emoji: '\u{26A1}', color: 'yellow' },
  cross_exchange_spread: { label: 'Cross-Exchange Spread', emoji: '\u{1F504}', color: 'green' },
  correlation: { label: 'Correlation Pairs', emoji: '\u{1F517}', color: 'purple' },
}

function formatTimeAgo(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  return `${hrs}h ${mins % 60}m ago`
}

function formatTimeLeft(isoString: string): string {
  const diff = new Date(isoString).getTime() - Date.now()
  if (diff <= 0) return 'expired'
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m`
  const hrs = Math.floor(mins / 60)
  return `${hrs}h ${mins % 60}m`
}

function ScoreBar({ score }: { score: number }) {
  const color = score >= 70 ? 'bg-green-500' : score >= 40 ? 'bg-yellow-500' : 'bg-gray-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
      <span className="text-xs text-gray-400">{score.toFixed(1)}</span>
    </div>
  )
}

function SummaryCard({ type, count, topScore }: {
  type: OpportunityTypeName
  count: number
  topScore: number
}) {
  const meta = TYPE_LABELS[type]
  const borderColor = {
    blue: 'border-blue-600',
    yellow: 'border-yellow-600',
    green: 'border-green-600',
    purple: 'border-purple-600',
  }[meta.color]

  return (
    <div className={`bg-gray-800 rounded-lg p-4 border-l-4 ${borderColor}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">{meta.label}</span>
      </div>
      <div className="text-2xl font-bold text-white">{count}</div>
      <div className="text-xs text-gray-400 mt-1">Top score: {topScore.toFixed(1)}</div>
    </div>
  )
}

function OpportunityTable({ opportunities, type }: {
  opportunities: ScannerOpportunity[]
  type: OpportunityTypeName
}) {
  if (opportunities.length === 0) {
    return <p className="text-gray-500 text-sm py-4 text-center">No opportunities found</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2 px-3">Symbol</th>
            <th className="text-left py-2 px-3">Score</th>
            <th className="text-left py-2 px-3">Key Metric</th>
            <th className="text-left py-2 px-3">Exchange</th>
            <th className="text-left py-2 px-3">Discovered</th>
            <th className="text-left py-2 px-3">TTL</th>
          </tr>
        </thead>
        <tbody>
          {opportunities.map((opp, i) => (
            <tr key={`${opp.symbol}-${i}`} className="border-b border-gray-700/50 hover:bg-gray-800/50">
              <td className="py-2 px-3 font-mono text-white">{opp.symbol}</td>
              <td className="py-2 px-3"><ScoreBar score={opp.score} /></td>
              <td className="py-2 px-3 text-gray-300">
                {type === 'funding_rate' && `${((opp.metrics.annualized_pct as number) || 0).toFixed(1)}% ann.`}
                {type === 'volatility' && `${((opp.metrics.change_pct as number) || 0).toFixed(1)}% 24h`}
                {type === 'cross_exchange_spread' && `${((opp.metrics.spread_pct as number) || 0).toFixed(3)}% spread`}
                {type === 'correlation' && `r=${((opp.metrics.correlation as number) || 0).toFixed(3)}`}
              </td>
              <td className="py-2 px-3 text-gray-400">{opp.source_exchange || '-'}</td>
              <td className="py-2 px-3 text-gray-400">{opp.discovered_at ? formatTimeAgo(opp.discovered_at) : '-'}</td>
              <td className="py-2 px-3 text-gray-400">{opp.expires_at ? formatTimeLeft(opp.expires_at) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Scanner() {
  const [data, setData] = useState<ScannerResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<OpportunityTypeName>('funding_rate')

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/scanner/opportunities`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [fetchData])

  if (loading) {
    return <div className="text-gray-400 py-8 text-center">Loading scanner data...</div>
  }

  if (error) {
    return <div className="text-red-400 py-8 text-center">Error: {error}</div>
  }

  const summary = data?.summary || {} as Record<OpportunityTypeName, { count: number; top_score: number; symbols: string[] }>
  const opportunities = data?.opportunities || {} as Record<OpportunityTypeName, ScannerOpportunity[]>

  const types: OpportunityTypeName[] = ['funding_rate', 'volatility', 'cross_exchange_spread', 'correlation']

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Token Scanner</h2>
        <button
          onClick={fetchData}
          className="px-3 py-1.5 rounded text-sm bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {types.map((t) => (
          <SummaryCard
            key={t}
            type={t}
            count={summary[t]?.count || 0}
            topScore={summary[t]?.top_score || 0}
          />
        ))}
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 border-b border-gray-700 pb-2">
        {types.map((t) => {
          const meta = TYPE_LABELS[t]
          const count = opportunities[t]?.length || 0
          return (
            <button
              key={t}
              onClick={() => setActiveTab(t)}
              className={`px-4 py-2 rounded-t text-sm transition-colors ${
                activeTab === t
                  ? 'bg-gray-800 text-white border-b-2 border-blue-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              {meta.label} ({count})
            </button>
          )
        })}
      </div>

      {/* Opportunity table */}
      <div className="bg-gray-800/50 rounded-lg p-4">
        <OpportunityTable
          opportunities={opportunities[activeTab] || []}
          type={activeTab}
        />
      </div>
    </div>
  )
}
