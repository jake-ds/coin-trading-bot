interface MetricCardProps {
  label: string
  value: string | number
  suffix?: string
  colorClass?: string
}

export default function MetricCard({ label, value, suffix, colorClass }: MetricCardProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <p className="text-sm text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorClass || 'text-white'}`}>
        {value}
        {suffix && <span className="text-sm font-normal text-gray-400 ml-1">{suffix}</span>}
      </p>
    </div>
  )
}
