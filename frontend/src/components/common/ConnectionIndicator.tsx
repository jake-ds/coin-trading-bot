interface ConnectionIndicatorProps {
  connected: boolean
}

function ConnectionIndicator({ connected }: ConnectionIndicatorProps) {
  return (
    <div className="flex items-center gap-1.5" title={connected ? 'Live (WebSocket)' : 'Disconnected'}>
      <span
        className={`inline-block w-2.5 h-2.5 rounded-full ${
          connected ? 'bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]' : 'bg-red-400'
        }`}
      />
      <span className="text-xs text-gray-400">
        {connected ? 'Live' : 'Offline'}
      </span>
    </div>
  )
}

export default ConnectionIndicator
