import type { Kline, SymbolInfo, JobSummary, JobsResponse } from '../types'

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${path}`)
  return r.json() as Promise<T>
}

export function fetchSymbols(): Promise<SymbolInfo[]> {
  return get<SymbolInfo[]>('/api/symbols')
}

export function fetchKlines(
  symbol: string,
  interval: string,
  limit = 1000,
): Promise<Kline[]> {
  const p = new URLSearchParams({ symbol, interval, limit: String(limit) })
  return get<Kline[]>(`/api/klines?${p}`)
}

export function fetchJobSummary(): Promise<JobSummary> {
  return get<JobSummary>('/api/jobs/summary')
}

export function fetchJobs(
  status?: string,
  limit = 50,
  offset = 0,
): Promise<JobsResponse> {
  const p = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (status) p.set('status', status)
  return get<JobsResponse>(`/api/jobs?${p}`)
}
