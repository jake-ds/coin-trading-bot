import { useState, useEffect, useCallback } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
  RadialBarChart,
  RadialBar,
} from 'recharts'
import apiClient from '../api/client'
import type {
  RiskPortfolioMetrics,
  DrawdownPoint,
  CorrelationReport,
} from '../api/types'

const SYMBOL_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#a855f7']

function varLevel(value: number | null, limit: number): 'safe' | 'warning' | 'danger' {
  if (value === null) return 'safe'
  const ratio = Math.abs(value) / limit
  if (ratio < 0.6) return 'safe'
  if (ratio < 0.85) return 'warning'
  return 'danger'
}

const levelColor = { safe: '#22c55e', warning: '#f59e0b', danger: '#ef4444' }

function GaugeCard({
  label,
  value,
  limit,
}: {
  label: string
  value: number | null
  limit: number
}) {
  const displayVal = value !== null ? Math.abs(value) : 0
  const pct = limit > 0 ? Math.min(displayVal / limit, 1) : 0
  const level = varLevel(value, limit)
  const data = [{ value: pct * 100, fill: levelColor[level] }]

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <p className="text-xs text-gray-500 mb-2 text-center">{label}</p>
      <div className="h-28">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="60%"
            outerRadius="90%"
            startAngle={180}
            endAngle={0}
            data={data}
            cx="50%"
            cy="80%"
          >
            <RadialBar dataKey="value" cornerRadius={4} background={{ fill: '#374151' }} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="text-center -mt-4">
        <span className={`text-lg font-mono font-bold`} style={{ color: levelColor[level] }}>
          {value !== null ? `${displayVal.toFixed(2)}%` : 'N/A'}
        </span>
        <span className="text-xs text-gray-500 ml-1">/ {limit}%</span>
      </div>
    </div>
  )
}

function RiskDashboard() {
  const [portfolio, setPortfolio] = useState<RiskPortfolioMetrics | null>(null)
  const [drawdownHistory, setDrawdownHistory] = useState<DrawdownPoint[]>([])
  const [correlation, setCorrelation] = useState<CorrelationReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      const [pRes, dRes, cRes] = await Promise.allSettled([
        apiClient.get<RiskPortfolioMetrics>('/risk/portfolio'),
        apiClient.get<{ history: DrawdownPoint[] }>('/risk/drawdown'),
        apiClient.get<CorrelationReport>('/risk/correlation'),
      ])

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (pRes.status === 'fulfilled' && !(pRes.value.data as any).error) {
        setPortfolio(pRes.value.data)
      }
      if (dRes.status === 'fulfilled') {
        setDrawdownHistory(dRes.value.data.history || [])
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (cRes.status === 'fulfilled' && !(cRes.value.data as any).error) {
        setCorrelation(cRes.value.data)
      }
      setError(null)
    } catch {
      setError('Failed to fetch risk data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 15000)
    return () => clearInterval(interval)
  }, [fetchData])

  if (loading && !portfolio) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Risk Dashboard</h2>
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4 w-1/3" />
          <div className="h-64 bg-gray-700 rounded" />
        </div>
      </div>
    )
  }

  if (error && !portfolio) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-4">Risk Dashboard</h2>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">{error}</div>
      </div>
    )
  }

  const varLimit = 5.0 // max_portfolio_var_pct default

  // Engine exposure stacked bar data
  const engineExposure =
    portfolio?.positions?.reduce(
      (acc, p) => {
        // Group by engine isn't available in positions, show by symbol
        acc.push({ name: p.symbol, value: p.value })
        return acc
      },
      [] as Array<{ name: string; value: number }>,
    ) || []

  // Symbol pie chart data
  const totalValue = engineExposure.reduce((s, e) => s + e.value, 0)
  const pieData = engineExposure.map((e, i) => ({
    name: e.name,
    value: e.value,
    fill: SYMBOL_COLORS[i % SYMBOL_COLORS.length],
  }))

  // Correlation heatmap
  const engines = ['funding_rate_arb', 'grid_trading', 'cross_exchange_arb', 'stat_arb']
  const corrMap = correlation?.cross_engine_correlations || {}

  // Concentration alerts
  const alerts = correlation?.alerts || []

  // Per-symbol concentration data
  const perSymbol = correlation?.per_symbol || {}
  const concentrationSymbols = Object.entries(perSymbol)
    .map(([sym, data]) => ({ symbol: sym, ...data }))
    .sort((a, b) => b.pct_of_capital - a.pct_of_capital)

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Risk Dashboard</h2>

      {/* VaR Gauge Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <GaugeCard label="Historical VaR" value={portfolio?.var_pct ?? null} limit={varLimit} />
        <GaugeCard label="Parametric VaR" value={portfolio?.parametric_var ?? null} limit={varLimit} />
        <GaugeCard label="Cornish-Fisher VaR" value={portfolio?.cornish_fisher_var ?? null} limit={varLimit} />
        <GaugeCard label="CVaR (ES)" value={portfolio?.cvar ?? null} limit={varLimit} />
      </div>

      {/* Portfolio Heat + Exposure Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-500 mb-1">Portfolio Heat</p>
          <div className="flex items-center gap-3">
            <div className="flex-1 bg-gray-700 rounded-full h-3">
              <div
                className="h-3 rounded-full transition-all"
                style={{
                  width: `${Math.min((portfolio?.heat ?? 0) * 100 / 0.15, 100)}%`,
                  backgroundColor:
                    (portfolio?.heat ?? 0) < 0.1 ? '#22c55e' : (portfolio?.heat ?? 0) < 0.13 ? '#f59e0b' : '#ef4444',
                }}
              />
            </div>
            <span className="text-sm font-mono text-gray-200">
              {((portfolio?.heat ?? 0) * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-gray-500">/ 15%</span>
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-500 mb-1">Exposure</p>
          <p className="text-lg font-mono font-semibold text-gray-100">
            {(portfolio?.exposure_pct ?? 0).toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500">{portfolio?.n_positions ?? 0} positions</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-500 mb-1">Stress VaR (MC)</p>
          <p className={`text-lg font-mono font-semibold ${
            portfolio?.stress_var != null ? (Math.abs(portfolio.stress_var) > varLimit ? 'text-red-400' : 'text-green-400') : 'text-gray-500'
          }`}>
            {portfolio?.stress_var != null ? `${Math.abs(portfolio.stress_var).toFixed(2)}%` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Drawdown Curve */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-6">
        <h3 className="text-lg font-bold mb-4">Drawdown Curve</h3>
        {drawdownHistory.length === 0 ? (
          <p className="text-gray-500 text-sm">No drawdown data yet. Data is recorded as PnL reports come in.</p>
        ) : (
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={drawdownHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#9ca3af"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v: string) => {
                    try {
                      return new Date(v).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    } catch {
                      return ''
                    }
                  }}
                />
                <YAxis stroke="#9ca3af" reversed domain={[0, 'auto']} tickFormatter={(v: number) => `${v}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#e5e7eb' }}
                  formatter={(value: number | undefined) => [`${(value ?? 0).toFixed(2)}%`, 'Drawdown']}
                  labelFormatter={(label: unknown) => {
                    try {
                      return new Date(String(label)).toLocaleString()
                    } catch {
                      return String(label)
                    }
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="drawdown_pct"
                  stroke="#ef4444"
                  fill="rgba(239, 68, 68, 0.2)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
        {drawdownHistory.length > 0 && (
          <div className="flex gap-6 mt-2 text-sm">
            <span className="text-gray-400">
              Current: <span className="text-red-400">{drawdownHistory[drawdownHistory.length - 1].drawdown_pct.toFixed(2)}%</span>
            </span>
            <span className="text-gray-400">
              Max: <span className="text-red-400">{Math.max(...drawdownHistory.map((d) => d.drawdown_pct)).toFixed(2)}%</span>
            </span>
          </div>
        )}
      </div>

      {/* Correlation Heatmap + Exposure Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Engine Correlation Heatmap */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold mb-4">Engine Correlation</h3>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr>
                  <th className="px-2 py-1" />
                  {engines.map((e) => (
                    <th key={e} className="px-2 py-1 text-gray-400 font-medium text-center">
                      {e.split('_').map((w) => w[0].toUpperCase()).join('')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {engines.map((row) => (
                  <tr key={row}>
                    <td className="px-2 py-1 text-gray-400 font-medium whitespace-nowrap">
                      {row.split('_').map((w) => w[0].toUpperCase()).join('')}
                    </td>
                    {engines.map((col) => {
                      if (row === col) {
                        return (
                          <td key={col} className="px-2 py-1 text-center">
                            <span className="inline-block w-8 h-8 leading-8 rounded bg-gray-600 text-gray-400 text-xs">1.0</span>
                          </td>
                        )
                      }
                      const key1 = `${row}|${col}`
                      const key2 = `${col}|${row}`
                      const data = corrMap[key1] || corrMap[key2]
                      const overlap = data?.overlap_pct ?? 0
                      const alpha = Math.round(overlap * 80 + 20)
                      const bg = overlap > 0 ? `rgba(239, 68, 68, ${alpha / 100})` : ''
                      return (
                        <td key={col} className="px-2 py-1 text-center">
                          <span
                            className="inline-block w-8 h-8 leading-8 rounded text-xs text-white/80"
                            style={{ backgroundColor: bg || '#374151' }}
                            title={`${row} ↔ ${col}: overlap ${(overlap * 100).toFixed(0)}%`}
                          >
                            {(overlap * 100).toFixed(0)}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Position Exposure Pie */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold mb-4">Position Exposure</h3>
          {pieData.length === 0 ? (
            <p className="text-gray-500 text-sm">No open positions.</p>
          ) : (
            <>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={70}
                      paddingAngle={2}
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                      formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, 'Value']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {pieData.map((d) => (
                  <span key={d.name} className="flex items-center gap-1 text-xs text-gray-400">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: d.fill }} />
                    {d.name} ({totalValue > 0 ? ((d.value / totalValue) * 100).toFixed(0) : 0}%)
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Engine Exposure Bar + Concentration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Engine Notional Bar */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold mb-4">Position by Symbol</h3>
          {engineExposure.length === 0 ? (
            <p className="text-gray-500 text-sm">No positions.</p>
          ) : (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={engineExposure} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" stroke="#9ca3af" tickFormatter={(v: number) => `$${v}`} />
                  <YAxis type="category" dataKey="name" stroke="#9ca3af" width={80} tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                    formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, 'Notional']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {engineExposure.map((_, i) => (
                      <Cell key={i} fill={SYMBOL_COLORS[i % SYMBOL_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Concentration Alerts */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold mb-4">Concentration Monitor</h3>
          {concentrationSymbols.length === 0 && alerts.length === 0 ? (
            <p className="text-gray-500 text-sm">No concentration data available.</p>
          ) : (
            <>
              {alerts.length > 0 && (
                <div className="mb-4 space-y-2">
                  {alerts.map((alert, i) => (
                    <div key={i} className="bg-red-500/10 border border-red-500/30 rounded px-3 py-2 text-xs text-red-400">
                      {alert}
                    </div>
                  ))}
                </div>
              )}
              <div className="space-y-2">
                {concentrationSymbols.slice(0, 10).map((s) => (
                  <div key={s.symbol} className="flex items-center gap-3">
                    <span className="text-xs text-gray-300 w-24 truncate">{s.symbol}</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${Math.min(s.pct_of_capital * 100 / 40, 100)}%`,
                          backgroundColor: s.pct_of_capital > 0.4 ? '#ef4444' : s.pct_of_capital > 0.3 ? '#f59e0b' : '#22c55e',
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono text-gray-400 w-10 text-right">
                      {(s.pct_of_capital * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default RiskDashboard
