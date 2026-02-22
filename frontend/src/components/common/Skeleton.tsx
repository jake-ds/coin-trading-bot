interface SkeletonProps {
  className?: string
}

export default function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse bg-gray-700 rounded ${className || 'h-4 w-full'}`}
    />
  )
}

export function MetricCardSkeleton() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <Skeleton className="h-3 w-20 mb-3" />
      <Skeleton className="h-7 w-24" />
    </div>
  )
}

export function PortfolioSummarySkeleton() {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <Skeleton className="h-3 w-32 mb-3" />
      <Skeleton className="h-9 w-40 mb-2" />
      <Skeleton className="h-3 w-20" />
    </div>
  )
}
