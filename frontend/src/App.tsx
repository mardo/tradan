import { useState } from 'react'
import { Chart } from './components/Chart'
import { JobStatus } from './components/JobStatus'
import { SymbolSelector } from './components/SymbolSelector'

type Tab = 'chart' | 'jobs'

function App() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [interval, setInterval] = useState('1h')
  const [tab, setTab] = useState<Tab>('chart')

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-4 py-2.5 flex items-center gap-4 shrink-0">
        <span className="text-base font-bold tracking-tight text-white">
          Tradan
        </span>

        {tab === 'chart' && (
          <SymbolSelector
            symbol={symbol}
            interval={interval}
            onChange={(s, i) => {
              setSymbol(s)
              setInterval(i)
            }}
          />
        )}

        <nav className="ml-auto flex gap-1">
          {(['chart', 'jobs'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-sm rounded capitalize transition-colors ${
                tab === t
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {t === 'jobs' ? 'Ingestion' : 'Chart'}
            </button>
          ))}
        </nav>
      </header>

      <main
        className={`flex-1 overflow-auto ${
          tab === 'chart' ? 'p-0' : 'p-6'
        }`}
      >
        {tab === 'chart' ? (
          <div className="h-[calc(100vh-49px)]">
            <Chart symbol={symbol} interval={interval} />
          </div>
        ) : (
          <JobStatus />
        )}
      </main>
    </div>
  )
}

export default App
