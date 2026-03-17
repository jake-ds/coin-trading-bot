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
    cash_value?: number
    unrealized_pnl?: number
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
  signal_score?: number
  signal_confidence?: number
}

export interface PositionsResponse {
  positions: Position[]
}

// Onchain Signal types
export interface SignalScore {
  name: string
  score: number       // -100 to +100
  confidence: number  // 0.0 to 1.0
  reason: string
}

export interface CompositeSignal {
  symbol: string
  action: 'BUY' | 'SELL' | 'HOLD'
  score: number       // -100 to +100
  confidence: number  // 0.0 to 1.0
  signals: SignalScore[]
  timestamp: string
}

export interface OnchainSignalsResponse {
  signals: Record<string, CompositeSignal>
}

export interface SignalHistoryEntry {
  timestamp: string
  signals: Record<string, CompositeSignal>
}

export interface SignalHistoryResponse {
  history: SignalHistoryEntry[]
}

export interface AuthStatusResponse {
  auth_enabled: boolean
  dev_mode: boolean
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface RefreshResponse {
  access_token: string
  token_type: string
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
    cash_value?: number
    unrealized_pnl?: number
  }
  metrics: Record<string, number>
  trades: Trade[]
  open_positions: Position[]
  emergency: { active: boolean }
  engine_performance: Record<string, {
    pnl: number
    win_rate: number
    total_trades: number
  }>
  onchain_signals: Record<string, CompositeSignal>
}

// Settings types
export interface SettingItem {
  key: string
  value: unknown
  default: unknown
  section: string
  description: string
  type: string
  requires_restart: boolean
  options?: string[]
}

export interface SettingsResponse {
  settings: SettingItem[]
}

export interface SettingsUpdateResponse {
  success: boolean
  changed: string[]
  previous: Record<string, unknown>
}

// Engine types
export type EngineStatusValue = 'stopped' | 'running' | 'paused' | 'error'

export interface EngineInfo {
  name: string
  description: string
  status: EngineStatusValue
  cycle_count: number
  total_pnl: number
  allocated_capital: number
  position_count: number
  max_positions: number
  loop_interval: number
  error: string | null
  role_ko?: string
  role_en?: string
  description_ko?: string
  key_params?: string
  symbols?: string[] | string[][]
}

export type EnginesResponse = Record<string, EngineInfo>

export interface EngineActionResponse {
  success: boolean
  engine: string
  action: string
}
