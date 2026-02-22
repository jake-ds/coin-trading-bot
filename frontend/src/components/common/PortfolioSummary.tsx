interface PortfolioSummaryProps {
  totalValue: number
  totalReturn: number
}

export default function PortfolioSummary({ totalValue, totalReturn }: PortfolioSummaryProps) {
  const returnColor = totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
  const returnSign = totalReturn >= 0 ? '+' : ''

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <p className="text-sm text-gray-400 mb-1">Total Portfolio Value</p>
      <p className="text-3xl font-bold text-white">
        ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </p>
      <p className={`text-sm mt-1 ${returnColor}`}>
        {returnSign}{totalReturn.toFixed(2)}% all time
      </p>
    </div>
  )
}
