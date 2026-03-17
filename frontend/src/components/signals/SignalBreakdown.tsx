import type { SignalScore } from '../../api/types'

interface SignalBreakdownProps {
  signals: SignalScore[]
}

function getScoreColor(score: number): string {
  if (score > 30) return 'text-green-400'
  if (score > 0) return 'text-green-400/70'
  if (score < -30) return 'text-red-400'
  if (score < 0) return 'text-red-400/70'
  return 'text-gray-400'
}

function getBarStyle(score: number): { width: string; bg: string; side: 'left' | 'right' } {
  const pct = Math.min(Math.abs(score), 100)
  if (score >= 0) {
    return {
      width: `${pct / 2}%`,
      bg: score > 30 ? 'bg-green-500' : 'bg-green-500/60',
      side: 'right',
    }
  }
  return {
    width: `${pct / 2}%`,
    bg: score < -30 ? 'bg-red-500' : 'bg-red-500/60',
    side: 'left',
  }
}

const SIGNAL_LABELS: Record<string, string> = {
  whale_flow: 'Whale Flow',
  sentiment: 'Sentiment',
  defi_flow: 'DeFi Flow',
  derivatives: 'Derivatives',
  market_trend: 'Market Trend',
}

export default function SignalBreakdown({ signals }: SignalBreakdownProps) {
  if (!signals || signals.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-gray-400 text-sm">
        No signal data available
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="text-left px-4 py-3 font-medium w-32">Signal</th>
            <th className="px-4 py-3 font-medium">Score</th>
            <th className="text-center px-4 py-3 font-medium w-20">Conf</th>
            <th className="text-left px-4 py-3 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((sig) => {
            const bar = getBarStyle(sig.score)
            return (
              <tr key={sig.name} className="border-b border-gray-700/50">
                <td className="px-4 py-3 text-white font-medium">
                  {SIGNAL_LABELS[sig.name] || sig.name}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-700 rounded-full relative">
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-600" />
                      {bar.side === 'right' ? (
                        <div
                          className={`absolute left-1/2 h-full ${bar.bg} rounded-r-full`}
                          style={{ width: bar.width }}
                        />
                      ) : (
                        <div
                          className={`absolute h-full ${bar.bg} rounded-l-full`}
                          style={{ right: '50%', width: bar.width }}
                        />
                      )}
                    </div>
                    <span className={`text-xs font-mono font-bold w-10 text-right ${getScoreColor(sig.score)}`}>
                      {sig.score > 0 ? '+' : ''}{sig.score.toFixed(0)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-center text-gray-300 text-xs">
                  {sig.confidence > 0 ? `${(sig.confidence * 100).toFixed(0)}%` : '--'}
                </td>
                <td className="px-4 py-3 text-gray-400 text-xs truncate max-w-xs" title={sig.reason}>
                  {sig.reason || '--'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
