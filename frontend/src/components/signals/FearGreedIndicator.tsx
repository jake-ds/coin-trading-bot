interface FearGreedIndicatorProps {
  value: number | null  // 0-100
  classification?: string
}

function getGradientColor(value: number): string {
  // 0 = extreme fear (red), 50 = neutral (yellow), 100 = extreme greed (green)
  if (value <= 25) return 'text-red-400'
  if (value <= 45) return 'text-orange-400'
  if (value <= 55) return 'text-yellow-400'
  if (value <= 75) return 'text-lime-400'
  return 'text-green-400'
}

function getBarGradient(value: number): string {
  if (value <= 25) return 'bg-red-500'
  if (value <= 45) return 'bg-orange-500'
  if (value <= 55) return 'bg-yellow-500'
  if (value <= 75) return 'bg-lime-500'
  return 'bg-green-500'
}

function getLabel(value: number, classification?: string): string {
  if (classification) return classification
  if (value <= 25) return 'Extreme Fear'
  if (value <= 45) return 'Fear'
  if (value <= 55) return 'Neutral'
  if (value <= 75) return 'Greed'
  return 'Extreme Greed'
}

export default function FearGreedIndicator({ value, classification }: FearGreedIndicatorProps) {
  if (value === null || value === undefined) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="text-sm text-gray-400 mb-2">Fear & Greed</div>
        <div className="text-gray-500 text-sm">No data</div>
      </div>
    )
  }

  const clampedValue = Math.max(0, Math.min(100, value))
  const colorClass = getGradientColor(clampedValue)
  const barClass = getBarGradient(clampedValue)
  const label = getLabel(clampedValue, classification)

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="text-sm text-gray-400 mb-2">Fear & Greed Index</div>
      <div className="flex items-center gap-3">
        <span className={`text-3xl font-bold ${colorClass}`}>{clampedValue}</span>
        <span className={`text-sm font-medium ${colorClass}`}>{label}</span>
      </div>
      {/* Bar */}
      <div className="mt-3 h-2 bg-gray-700 rounded-full relative overflow-hidden">
        <div
          className={`absolute left-0 h-full ${barClass} rounded-full transition-all`}
          style={{ width: `${clampedValue}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Fear</span>
        <span>Greed</span>
      </div>
    </div>
  )
}
