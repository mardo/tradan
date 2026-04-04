import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  ColorType,
  CandlestickSeries,
  type IChartApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import { fetchKlines } from '../api/client'

interface Props {
  symbol: string
  interval: string
}

export function Chart({ symbol, interval }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#030712' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      timeScale: { timeVisible: true, secondsVisible: false },
    })

    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    chartRef.current = chart

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }
    window.addEventListener('resize', handleResize)

    let cancelled = false
    setLoading(true)
    setError(null)

    fetchKlines(symbol, interval, 1000)
      .then((data) => {
        if (cancelled) return
        series.setData(
          data.map((d) => ({
            time: Math.floor(d.time / 1000) as UTCTimestamp,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
          })),
        )
        chart.timeScale().fitContent()
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
    }
  }, [symbol, interval])

  return (
    <div className="relative w-full h-full">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/60 z-10">
          <span className="text-gray-400 text-sm">Loading…</span>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-red-400 text-sm">{error}</span>
        </div>
      )}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}
