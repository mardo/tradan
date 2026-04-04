export interface Kline {
  /** open_time in milliseconds (divide by 1000 for lightweight-charts UTCTimestamp) */
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  num_trades: number
}

export interface SymbolInfo {
  symbol: string
  intervals: string[]
}

export interface JobSummary {
  pending?: number
  running?: number
  done?: number
  failed?: number
}

export interface Job {
  id: number
  symbol: string
  interval: string
  year: number
  month: number
  status: string
  claimed_at: string | null
  completed_at: string | null
  error: string | null
}

export interface JobsResponse {
  total: number
  jobs: Job[]
}
