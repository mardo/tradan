import { useEffect, useState } from 'react'
import { fetchSymbols } from '../api/client'
import type { SymbolInfo } from '../types'

interface Props {
  symbol: string
  interval: string
  onChange: (symbol: string, interval: string) => void
}

export function SymbolSelector({ symbol, interval, onChange }: Props) {
  const [symbols, setSymbols] = useState<SymbolInfo[]>([])

  useEffect(() => {
    fetchSymbols().then(setSymbols).catch(console.error)
  }, [])

  const current = symbols.find((s) => s.symbol === symbol)

  return (
    <div className="flex items-center gap-2">
      <select
        value={symbol}
        onChange={(e) => {
          const s = symbols.find((x) => x.symbol === e.target.value)
          onChange(e.target.value, s?.intervals[0] ?? interval)
        }}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
      >
        {symbols.map((s) => (
          <option key={s.symbol} value={s.symbol}>
            {s.symbol}
          </option>
        ))}
      </select>

      <select
        value={interval}
        onChange={(e) => onChange(symbol, e.target.value)}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 focus:outline-none focus:border-indigo-500"
      >
        {(current?.intervals ?? []).map((i) => (
          <option key={i} value={i}>
            {i}
          </option>
        ))}
      </select>
    </div>
  )
}
