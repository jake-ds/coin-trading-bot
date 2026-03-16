interface PortfolioSummaryProps {
  totalValue: number
  totalReturn: number
  cashValue?: number
  unrealizedPnl?: number
}

export default function PortfolioSummary({ totalValue, totalReturn, cashValue, unrealizedPnl }: PortfolioSummaryProps) {
  const returnColor = totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
  const returnSign = totalReturn >= 0 ? '+' : ''
  const pnlColor = (unrealizedPnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
  const pnlSign = (unrealizedPnl ?? 0) >= 0 ? '+' : ''

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <p className="text-sm text-gray-400 mb-1">Total Portfolio Value</p>
      <p className="text-3xl font-bold text-white">
        ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </p>
      {cashValue != null && unrealizedPnl != null ? (
        <div className="flex gap-4 mt-2 text-sm">
          <span className="text-gray-400">
            Cash: ${cashValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className={pnlColor}>
            Unrealized PnL: {pnlSign}{unrealizedPnl.toFixed(2)} USDT
          </span>
        </div>
      ) : (
        <p className={`text-sm mt-1 ${returnColor}`}>
          {returnSign}{totalReturn.toFixed(2)}% all time
        </p>
      )}
    </div>
  )
}
