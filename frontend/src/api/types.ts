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
  wins: number
  losses: number
  avg_pnl: number
  consecutive_losses: number
  active: boolean
  sharpe_ratio: number
  profit_factor: number
  disabled: boolean
  disabled_reason: string | null
  pnl_history: number[]
}

export interface ToggleResponse {
  name: string
  active: boolean
  success: boolean
  has_open_positions?: boolean
  open_position_count?: number
  warning?: string
}

export interface RegimeResponse {
  regime: string | null
}

export interface AnalyticsResponse {
  equity_curve: EquityPoint[]
  drawdown: DrawdownPoint[]
  trade_markers: TradeMarker[]
  monthly_returns: MonthlyReturn[]
  stats: AnalyticsStats
  range: string
}

export interface EquityPoint {
  timestamp: string
  total_value: number
}

export interface DrawdownPoint {
  timestamp: string
  drawdown_pct: number
}

export interface TradeMarker {
  timestamp: string
  value: number
  side: 'BUY' | 'SELL'
  symbol: string
  price: number
}

export interface MonthlyReturn {
  month: string
  return_pct: number
}

export interface AnalyticsStats {
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown_pct: number
  profit_factor: number
  avg_trade_pnl: number
  best_trade: number
  worst_trade: number
  total_return_pct: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
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
