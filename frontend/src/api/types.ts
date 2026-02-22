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
  total: number
  page: number
  limit: number
  total_pages: number
}

export interface Trade {
  timestamp: string
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  price: number
  pnl?: number
  strategy?: string
}

export interface Position {
  symbol: string
  quantity: number
  entry_price: number
  current_price: number
  unrealized_pnl: number
  stop_loss: number
  take_profit: number
  opened_at?: string
  strategy?: string
}

export interface PositionsResponse {
  positions: Position[]
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

export interface WsStatusPayload {
  status: string
  started_at: string | null
  cycle_metrics: {
    cycle_count: number
    average_cycle_duration: number
    last_cycle_time: number | null
  }
  portfolio: {
    balances: Record<string, number>
    positions: unknown[]
    total_value: number
  }
  metrics: Record<string, number>
  regime: string | null
  trades: Trade[]
  strategy_stats: Record<string, StrategyStats>
  open_positions: Position[]
}
