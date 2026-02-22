import { useEffect, useState, useCallback } from 'react'
import apiClient from '../api/client'
import type { StatusResponse, PortfolioResponse, MetricsResponse, RegimeResponse } from '../api/types'
import StatusBadge from '../components/common/StatusBadge'
import MetricCard from '../components/common/MetricCard'
import RegimeBadge from '../components/common/RegimeBadge'
import PortfolioSummary from '../components/common/PortfolioSummary'
import { MetricCardSkeleton, PortfolioSummarySkeleton } from '../components/common/Skeleton'

interface DashboardData {
  status: StatusResponse | null
  portfolio: PortfolioResponse | null
  metrics: MetricsResponse | null
  regime: RegimeResponse | null
}

function Dashboard() {
  const [data, setData] = useState<DashboardData>({
    status: null,
    portfolio: null,
    metrics: null,
    regime: null,
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, portfolioRes, metricsRes, regimeRes] = await Promise.allSettled([
        apiClient.get<StatusResponse>('/status'),
        apiClient.get<PortfolioResponse>('/portfolio'),
        apiClient.get<MetricsResponse>('/metrics'),
        apiClient.get<RegimeResponse>('/regime'),
      ])

      setData({
        status: statusRes.status === 'fulfilled' ? statusRes.value.data : null,
        portfolio: portfolioRes.status === 'fulfilled' ? portfolioRes.value.data : null,
        metrics: metricsRes.status === 'fulfilled' ? metricsRes.value.data : null,
        regime: regimeRes.status === 'fulfilled' ? regimeRes.value.data : null,
      })
      setError(null)
    } catch {
      setError('Failed to fetch dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [fetchData])

  const metrics = data.metrics?.metrics || {}
  const portfolio = data.portfolio?.portfolio
  const cycleMetrics = data.status?.cycle_metrics

  if (loading) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Dashboard</h2>
        </div>
        <div className="mb-6">
          <PortfolioSummarySkeleton />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <MetricCardSkeleton key={i} />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      </div>
    )
  }

  const totalReturn = Number(metrics.total_return_pct || 0)
  const winRate = Number(metrics.win_rate || 0)
  const totalTrades = Number(metrics.total_trades || 0)
  const maxDrawdown = Number(metrics.max_drawdown_pct || 0)
  const totalValue = portfolio?.total_value || 0

  const winRateColor = winRate >= 50 ? 'text-green-400' : winRate > 0 ? 'text-amber-400' : 'text-gray-400'
  const drawdownColor = maxDrawdown > 10 ? 'text-red-400' : maxDrawdown > 5 ? 'text-amber-400' : 'text-green-400'

  const formatCycleTime = (t: number | null | undefined): string => {
    if (t == null) return '--'
    const d = new Date(t * 1000)
    return d.toLocaleTimeString()
  }

  const formatDuration = (s: number | undefined): string => {
    if (s == null || s === 0) return '--'
    return `${s.toFixed(1)}s`
  }

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <div className="flex items-center gap-3">
          <RegimeBadge regime={data.regime?.regime ?? null} />
          <StatusBadge status={data.status?.status || 'stopped'} />
        </div>
      </div>

      {/* Portfolio summary */}
      <div className="mb-6">
        <PortfolioSummary totalValue={totalValue} totalReturn={totalReturn} />
      </div>

      {/* Metric cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <MetricCard
          label="Total Return"
          value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}`}
          suffix="%"
          colorClass={totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MetricCard
          label="Win Rate"
          value={winRate.toFixed(1)}
          suffix="%"
          colorClass={winRateColor}
        />
        <MetricCard
          label="Total Trades"
          value={totalTrades}
        />
        <MetricCard
          label="Max Drawdown"
          value={maxDrawdown.toFixed(2)}
          suffix="%"
          colorClass={drawdownColor}
        />
        <MetricCard
          label="Last Cycle"
          value={formatCycleTime(cycleMetrics?.last_cycle_time)}
        />
        <MetricCard
          label="Avg Cycle Duration"
          value={formatDuration(cycleMetrics?.average_cycle_duration)}
        />
      </div>

      {/* Cycle info */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-2">Cycle Info</h3>
        <div className="flex flex-wrap gap-6 text-sm">
          <div>
            <span className="text-gray-500">Cycles: </span>
            <span className="text-white font-medium">{cycleMetrics?.cycle_count ?? 0}</span>
          </div>
          <div>
            <span className="text-gray-500">Avg Duration: </span>
            <span className="text-white font-medium">
              {formatDuration(cycleMetrics?.average_cycle_duration)}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Last Cycle: </span>
            <span className="text-white font-medium">
              {formatCycleTime(cycleMetrics?.last_cycle_time)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
