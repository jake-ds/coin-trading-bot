import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Legend } from 'recharts'
import type { SignalHistoryEntry } from '../../api/types'

interface SignalHistoryChartProps {
  history: SignalHistoryEntry[]
  symbol: string
}

const SYMBOL_COLORS: Record<string, string> = {
  'BTC/USDT': '#f7931a',
  'ETH/USDT': '#627eea',
  'SOL/USDT': '#9945ff',
  'BNB/USDT': '#f3ba2f',
  'XRP/USDT': '#00aae4',
  'DOGE/USDT': '#c2a633',
  'ADA/USDT': '#0033ad',
  'AVAX/USDT': '#e84142',
  'LINK/USDT': '#2a5ada',
  'DOT/USDT': '#e6007a',
  'MATIC/USDT': '#8247e5',
  'SUI/USDT': '#4da2ff',
  'NEAR/USDT': '#00c1de',
  'ARB/USDT': '#28a0f0',
  'OP/USDT': '#ff0420',
  'APT/USDT': '#06c9a1',
  'PEPE/USDT': '#4c9141',
  'UNI/USDT': '#ff007a',
  'ATOM/USDT': '#2e3148',
  'FIL/USDT': '#0090ff',
  'LTC/USDT': '#bfbbbb',
  'TRX/USDT': '#ff0013',
  'WIF/USDT': '#8b5e3c',
  'AAVE/USDT': '#b6509e',
  'RENDER/USDT': '#1ad9e0',
}

/** Resolve color: try exact key, then strip :USDT suffix */
function getSymbolColor(symbol: string): string {
  return SYMBOL_COLORS[symbol]
    || SYMBOL_COLORS[symbol.replace(':USDT', '')]
    || '#3b82f6'
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

export default function SignalHistoryChart({ history, symbol }: SignalHistoryChartProps) {
  if (!history || history.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center text-gray-400">
        No signal history available yet
      </div>
    )
  }

  // Build chart data: each point has timestamp + score for the selected symbol
  const chartData = history
    .filter((entry) => entry.signals[symbol])
    .map((entry) => ({
      time: formatTime(entry.timestamp),
      timestamp: entry.timestamp,
      score: entry.signals[symbol]?.score ?? 0,
    }))

  if (chartData.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center text-gray-400">
        No history for {symbol}
      </div>
    )
  }

  const lineColor = getSymbolColor(symbol)

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="time"
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[-100, 100]}
            stroke="#6b7280"
            tick={{ fontSize: 11 }}
            ticks={[-100, -50, 0, 50, 100]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Legend />
          <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="5 5" label={{ value: 'Buy +30', fill: '#22c55e', fontSize: 10 }} />
          <ReferenceLine y={-30} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Sell -30', fill: '#ef4444', fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#4b5563" />
          <Line
            type="monotone"
            dataKey="score"
            stroke={lineColor}
            strokeWidth={2}
            dot={false}
            name={symbol.replace('/USDT', '').replace(':USDT', '')}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
