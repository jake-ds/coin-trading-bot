interface StatusBadgeProps {
  status: string
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const isRunning = status.toLowerCase() === 'running'

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold ${
        isRunning
          ? 'bg-green-500/20 text-green-400'
          : 'bg-red-500/20 text-red-400'
      }`}
    >
      <span
        className={`w-2 h-2 rounded-full ${
          isRunning ? 'bg-green-400 animate-pulse' : 'bg-red-400'
        }`}
      />
      {status.toUpperCase()}
    </span>
  )
}
