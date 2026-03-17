import type { CompositeSignal } from '../../api/types'

interface SignalGaugeProps {
  symbol: string
  signal: CompositeSignal
  compact?: boolean
}

function getActionColor(action: string): { bg: string; text: string } {
  switch (action) {
    case 'BUY':
      return { bg: 'bg-green-500/20', text: 'text-green-400' }
    case 'SELL':
      return { bg: 'bg-red-500/20', text: 'text-red-400' }
    default:
      return { bg: 'bg-gray-500/20', text: 'text-gray-400' }
  }
}

function getBarColor(score: number): string {
  if (score > 30) return 'bg-green-500'
  if (score > 0) return 'bg-green-500/60'
  if (score < -30) return 'bg-red-500'
  if (score < 0) return 'bg-red-500/60'
  return 'bg-gray-500'
}

export default function SignalGauge({ symbol, signal, compact = false }: SignalGaugeProps) {
  const { bg, text } = getActionColor(signal.action)
  const score = signal.score
  // Map -100..+100 to 0..100% for the bar
  const barPct = Math.abs(score)
  const isPositive = score >= 0

  const displaySymbol = symbol.replace('/USDT', '').replace(':USDT', '')

  if (compact) {
    return (
      <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 flex items-center gap-3 min-w-[160px]">
        <span className="text-white font-bold text-sm">{displaySymbol}</span>
        <div className="flex-1 h-2 bg-gray-700 rounded-full relative overflow-hidden">
          {isPositive ? (
            <div className="absolute left-1/2 h-full rounded-r-full" style={{ width: `${barPct / 2}%` }}>
              <div className={`h-full ${getBarColor(score)} rounded-r-full`} />
            </div>
          ) : (
            <div className="absolute h-full rounded-l-full" style={{ right: '50%', width: `${barPct / 2}%` }}>
              <div className={`h-full ${getBarColor(score)} rounded-l-full`} />
            </div>
          )}
        </div>
        <span className={`text-xs font-bold ${text}`}>
          {score > 0 ? '+' : ''}{score.toFixed(0)}
        </span>
        <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${bg} ${text}`}>
          {signal.action}
        </span>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <span className="text-white font-bold">{displaySymbol}</span>
        <span className={`text-xs font-bold px-2 py-1 rounded ${bg} ${text}`}>
          {signal.action}
        </span>
      </div>

      {/* Score bar */}
      <div className="relative h-3 bg-gray-700 rounded-full mb-2">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-500" />
        {isPositive ? (
          <div
            className={`absolute left-1/2 h-full ${getBarColor(score)} rounded-r-full transition-all`}
            style={{ width: `${barPct / 2}%` }}
          />
        ) : (
          <div
            className={`absolute h-full ${getBarColor(score)} rounded-l-full transition-all`}
            style={{ right: '50%', width: `${barPct / 2}%` }}
          />
        )}
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>-100</span>
        <span className={`font-bold ${text}`}>
          {score > 0 ? '+' : ''}{score.toFixed(1)}
        </span>
        <span>+100</span>
      </div>

      <div className="mt-2 text-xs text-gray-400">
        Confidence: <span className="text-white">{(signal.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}
