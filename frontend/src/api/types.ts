export interface StatusResponse {
  status: string
  started_at: string | null
  cycle_metrics: {
    cycle_count: number
    average_cycle_duration: number
    last_cycle_time: number | null
  }
}

export interface PortfolioResponse {
  portfolio: {
    balances: Record<string, number>
    positions: unknown[]
    total_value: number
  }
}

export interface MetricsResponse {
  metrics: Record<string, number>
}

export interface TradesResponse {
  trades: Trade[]
}

export interface Trade {
  timestamp: string
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  price: number
}

export interface StrategiesResponse {
  strategies: Record<string, StrategyStats>
}

export interface StrategyStats {
  total_pnl: number
  win_rate: number
  total_trades: number
  consecutive_losses: number
  active: boolean
  sharpe_ratio?: number
}

export interface RegimeResponse {
  regime: string | null
}
