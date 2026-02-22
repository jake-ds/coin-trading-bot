interface RegimeBadgeProps {
  regime: string | null
}

function getRegimeStyle(regime: string | null): { bg: string; text: string; label: string } {
  if (!regime) {
    return { bg: 'bg-gray-500/20', text: 'text-gray-400', label: 'UNKNOWN' }
  }
  const r = regime.toUpperCase()
  if (r.includes('TRENDING_UP')) {
    return { bg: 'bg-green-500/20', text: 'text-green-400', label: 'TRENDING UP' }
  }
  if (r.includes('TRENDING_DOWN')) {
    return { bg: 'bg-red-500/20', text: 'text-red-400', label: 'TRENDING DOWN' }
  }
  if (r.includes('RANGING')) {
    return { bg: 'bg-amber-500/20', text: 'text-amber-400', label: 'RANGING' }
  }
  if (r.includes('VOLATIL')) {
    return { bg: 'bg-purple-500/20', text: 'text-purple-400', label: 'VOLATILE' }
  }
  return { bg: 'bg-gray-500/20', text: 'text-gray-400', label: regime.toUpperCase() }
}

export default function RegimeBadge({ regime }: RegimeBadgeProps) {
  const style = getRegimeStyle(regime)

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold ${style.bg} ${style.text}`}
    >
      {style.label}
    </span>
  )
}
