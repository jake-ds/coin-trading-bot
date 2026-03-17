import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts'
import type { CompositeSignal } from '../../api/types'

interface SignalOverviewChartProps {
  signals: Record<string, CompositeSignal>
}

function getBarColor(score: number): string {
  if (score >= 50) return '#22c55e'
  if (score >= 30) return '#4ade80'
  if (score > 0) return '#86efac'
  if (score === 0) return '#6b7280'
  if (score > -30) return '#fca5a5'
  if (score > -50) return '#f87171'
  return '#ef4444'
}

export default function SignalOverviewChart({ signals }: SignalOverviewChartProps) {
  const entries = Object.entries(signals)
  if (entries.length === 0) return null

  const chartData = entries
    .map(([symbol, sig]) => ({
      symbol: symbol.replace('/USDT', '').replace(':USDT', ''),
      score: sig.score,
      action: sig.action,
      confidence: sig.confidence,
      fullSymbol: symbol,
    }))
    .sort((a, b) => b.score - a.score)

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28 + 40)}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis
            type="number"
            domain={[-100, 100]}
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            ticks={[-100, -50, 0, 50, 100]}
          />
          <YAxis
            type="category"
            dataKey="symbol"
            stroke="#6b7280"
            tick={{ fontSize: 11, fill: '#d1d5db' }}
            width={55}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(value: any, _name: any, props: any) => {
              const v = Number(value) || 0
              const action = props?.payload?.action ?? ''
              const conf = props?.payload?.confidence ?? 0
              return [`${v > 0 ? '+' : ''}${v.toFixed(0)} (${action}, conf: ${(conf * 100).toFixed(0)}%)`, 'Score']
            }}
          />
          <ReferenceLine x={30} stroke="#22c55e" strokeDasharray="3 3" />
          <ReferenceLine x={-30} stroke="#ef4444" strokeDasharray="3 3" />
          <ReferenceLine x={0} stroke="#4b5563" />
          <Bar dataKey="score" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {chartData.map((entry, idx) => (
              <Cell key={idx} fill={getBarColor(entry.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
