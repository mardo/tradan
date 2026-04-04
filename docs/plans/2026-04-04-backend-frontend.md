# Backend + Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `backend` Python package with a FastAPI HTTP API and build a React + TypeScript + Tailwind + Vite frontend that displays candlestick charts and ingestion job status, wired together with a single root `pnpm dev` command.

**Architecture:** The `backend` package gains a `api.py` FastAPI app that exposes read-only endpoints over the existing PostgreSQL tables (`klines`, `ingest_jobs`). The Vite dev server proxies all `/api/*` requests to the backend on port 8000, so no CORS issues in development and no hardcoded ports in frontend code. Both services are started in parallel from the monorepo root via `concurrently`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, psycopg3 (existing), React 18, TypeScript, Tailwind CSS v4 (`@tailwindcss/vite`), lightweight-charts v4, Vite 6, pnpm (monorepo), concurrently.

---

## File Map

**Modified:**
- `backend/pyproject.toml` — add `fastapi`, `uvicorn[standard]` dependencies
- `backend/src/ingester/db.py` — add `get_conn()` context-manager helper
- `tradan/package.json` — add `dev`, `dev:backend`, `dev:frontend` scripts + `concurrently` devDependency

**Created (backend):**
- `backend/src/ingester/api.py` — FastAPI app, CORS middleware, router mounting, `/api/health`
- `backend/src/ingester/routers/__init__.py` — empty package marker
- `backend/src/ingester/routers/klines.py` — `GET /api/klines`, `GET /api/symbols`
- `backend/src/ingester/routers/jobs.py` — `GET /api/jobs/summary`, `GET /api/jobs`

**Created (frontend — all paths relative to repo root):**
- `frontend/` — Vite React TypeScript project scaffold
- `frontend/vite.config.ts` — Tailwind plugin + `/api` proxy to `http://localhost:8000`
- `frontend/src/index.css` — `@import "tailwindcss"` (Tailwind v4 entry point)
- `frontend/src/types.ts` — shared TypeScript interfaces
- `frontend/src/api/client.ts` — typed `fetch` wrappers
- `frontend/src/components/Chart.tsx` — lightweight-charts candlestick
- `frontend/src/components/SymbolSelector.tsx` — symbol + interval `<select>` dropdowns
- `frontend/src/components/JobStatus.tsx` — summary cards + paginated job table
- `frontend/src/App.tsx` — top-level layout with header and tab navigation
- `frontend/src/main.tsx` — React entry point (minor edit to import CSS)

---

## Task 1: Add FastAPI deps to ingester

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add dependencies**

```bash
cd backend
uv add fastapi "uvicorn[standard]"
```

Expected: `pyproject.toml` dependencies section now includes `fastapi` and `uvicorn[standard]`. Lock file updated.

- [ ] **Step 2: Verify install**

```bash
uv run python -c "import fastapi, uvicorn; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/pyproject.toml backend/uv.lock
git commit -m "feat: add fastapi + uvicorn to ingester"
```

---

## Task 2: Add `get_conn` context manager to db.py

**Files:**
- Modify: `backend/src/ingester/db.py`

`psycopg3.Connection` does not auto-close when used as a context manager — add a helper that connects, yields, and always closes.

- [ ] **Step 1: Add imports at top of `db.py`**

After `import os` (first import block), add:

```python
from contextlib import contextmanager
from typing import Generator
```

- [ ] **Step 2: Add `get_conn` after the `connect()` function**

Insert immediately after the existing `connect()` function (around line 18):

```python
@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    """Yield an open connection and close it on exit (commit/rollback handled by caller)."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 3: Verify**

```bash
cd backend
uv run python -c "
from ingester.db import get_conn
with get_conn() as c:
    print(c.execute('SELECT 1').fetchone())
"
```

Expected output: `(1,)`

- [ ] **Step 4: Commit**

```bash
cd ..
git add backend/src/ingester/db.py
git commit -m "feat: add get_conn context manager to db"
```

---

## Task 3: Create FastAPI app (`api.py`)

**Files:**
- Create: `backend/src/ingester/api.py`
- Create: `backend/src/ingester/routers/__init__.py`

- [ ] **Step 1: Create empty routers package**

```bash
touch backend/src/ingester/routers/__init__.py
```

- [ ] **Step 2: Create `api.py`**

`backend/src/ingester/api.py`:

```python
"""
FastAPI application.

Start with:
  cd backend && uv run uvicorn ingester.api:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tradan Ingester API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Routers are imported here (after they exist — see Tasks 4 & 5)
from .routers import klines as klines_router  # noqa: E402
from .routers import jobs as jobs_router      # noqa: E402

app.include_router(klines_router.router, prefix="/api")
app.include_router(jobs_router.router, prefix="/api")
```

> **Note:** The router imports will fail until Tasks 4 and 5 are done. Create stub files first (next step).

- [ ] **Step 3: Create stub router files so `api.py` can be imported**

`backend/src/ingester/routers/klines.py` (stub):

```python
from fastapi import APIRouter
router = APIRouter()
```

`backend/src/ingester/routers/jobs.py` (stub):

```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 4: Verify app starts**

```bash
cd backend
uv run uvicorn ingester.api:app --port 8000 &
sleep 2
curl -s http://localhost:8000/api/health
kill %1
```

Expected output: `{"status":"ok"}`

- [ ] **Step 5: Commit**

```bash
cd ..
git add backend/src/ingester/api.py ingester/src/ingester/routers/
git commit -m "feat: scaffold FastAPI app with health endpoint"
```

---

## Task 4: Klines router

**Files:**
- Modify: `backend/src/ingester/routers/klines.py`

- [ ] **Step 1: Implement klines router**

Replace the stub content of `backend/src/ingester/routers/klines.py` with:

```python
"""
Endpoints:
  GET /api/symbols                            – distinct symbols and their intervals
  GET /api/klines?symbol=&interval=&limit=&from_time=&to_time=
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..db import get_conn

router = APIRouter()


@router.get("/symbols")
def get_symbols() -> list[dict]:
    """Return each distinct symbol with its available intervals."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol, interval FROM klines ORDER BY symbol, interval"
        ).fetchall()

    result: dict[str, list[str]] = {}
    for symbol, interval in rows:
        result.setdefault(symbol, []).append(interval)
    return [{"symbol": k, "intervals": v} for k, v in result.items()]


@router.get("/klines")
def get_klines(
    symbol: str = Query(..., description="e.g. BTCUSDT"),
    interval: str = Query(..., description="e.g. 1m, 1h, 1d"),
    limit: int = Query(1000, ge=1, le=5000),
    from_time: Optional[int] = Query(None, description="open_time >= this value (ms epoch)"),
    to_time: Optional[int] = Query(None, description="open_time <= this value (ms epoch)"),
) -> list[dict]:
    """
    Return candlestick rows sorted ascending by open_time.
    `time` field is open_time in milliseconds (convert to seconds for lightweight-charts).
    """
    conditions = ["symbol = %s", "interval = %s"]
    params: list = [symbol, interval]

    if from_time is not None:
        conditions.append("open_time >= %s")
        params.append(from_time)
    if to_time is not None:
        conditions.append("open_time <= %s")
        params.append(to_time)

    where = " AND ".join(conditions)

    with get_conn() as conn:
        # Fetch newest first to honour LIMIT, then reverse for ascending order
        rows = conn.execute(
            f"""
            SELECT open_time, open, high, low, close, volume, num_trades
            FROM   klines
            WHERE  {where}
            ORDER  BY open_time DESC
            LIMIT  %s
            """,
            params + [limit],
        ).fetchall()

    return [
        {
            "time": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "num_trades": r[6],
        }
        for r in reversed(rows)
    ]
```

- [ ] **Step 2: Verify**

```bash
cd backend
uv run uvicorn ingester.api:app --port 8000 &
sleep 2
curl -s "http://localhost:8000/api/symbols" | python3 -m json.tool | head -20
curl -s "http://localhost:8000/api/klines?symbol=BTCUSDT&interval=1h&limit=3" | python3 -m json.tool
kill %1
```

Expected: JSON array of symbols for the first call; 3 kline objects for the second.

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/src/ingester/routers/klines.py
git commit -m "feat: add /api/symbols and /api/klines endpoints"
```

---

## Task 5: Jobs router

**Files:**
- Modify: `backend/src/ingester/routers/jobs.py`

- [ ] **Step 1: Implement jobs router**

Replace stub content of `backend/src/ingester/routers/jobs.py`:

```python
"""
Endpoints:
  GET /api/jobs/summary          – status counts {pending: N, done: N, ...}
  GET /api/jobs?status=&symbol=&limit=50&offset=0
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..db import get_conn

router = APIRouter()


@router.get("/jobs/summary")
def get_jobs_summary() -> dict:
    """Return a count per status value."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT status, count(*) FROM ingest_jobs GROUP BY status ORDER BY status"
        ).fetchall()
    return {row[0]: row[1] for row in rows}


@router.get("/jobs")
def get_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Return paginated job rows plus a total count."""
    conditions: list[str] = []
    params: list = []

    if status:
        conditions.append("status = %s")
        params.append(status)
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_conn() as conn:
        total: int = conn.execute(
            f"SELECT count(*) FROM ingest_jobs {where}", params
        ).fetchone()[0]  # type: ignore[index]

        rows = conn.execute(
            f"""
            SELECT id, symbol, interval, year, month,
                   status, claimed_at, completed_at, error
            FROM   ingest_jobs
            {where}
            ORDER  BY id DESC
            LIMIT  %s OFFSET %s
            """,
            params + [limit, offset],
        ).fetchall()

    return {
        "total": total,
        "jobs": [
            {
                "id": r[0],
                "symbol": r[1],
                "interval": r[2],
                "year": r[3],
                "month": r[4],
                "status": r[5],
                "claimed_at": r[6].isoformat() if r[6] else None,
                "completed_at": r[7].isoformat() if r[7] else None,
                "error": r[8],
            }
            for r in rows
        ],
    }
```

- [ ] **Step 2: Verify**

```bash
cd backend
uv run uvicorn ingester.api:app --port 8000 &
sleep 2
curl -s "http://localhost:8000/api/jobs/summary" | python3 -m json.tool
curl -s "http://localhost:8000/api/jobs?limit=2" | python3 -m json.tool
kill %1
```

Expected: `{"done": N, "failed": N, "pending": N}` for summary; `{total: N, jobs: [...]}` for jobs.

- [ ] **Step 3: Commit**

```bash
cd ..
git add backend/src/ingester/routers/jobs.py
git commit -m "feat: add /api/jobs/summary and /api/jobs endpoints"
```

---

## Task 6: Create Vite + React + TypeScript frontend scaffold

**Files:**
- Create: `frontend/` (entire directory)

- [ ] **Step 1: Scaffold with Vite**

Run from the repo root (`tradan/`):

```bash
npm create vite@latest frontend -- --template react-ts
```

When prompted, confirm creation in the `frontend/` directory.

- [ ] **Step 2: Install dependencies**

```bash
cd frontend
npm install
npm install lightweight-charts
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 3: Verify dev server starts (then stop it)**

```bash
npm run dev &
sleep 3
curl -s http://localhost:5173 | head -5
kill %1
```

Expected: HTML output with `<title>Vite + React + TS</title>` or similar.

- [ ] **Step 4: Commit scaffold**

```bash
cd ..
git add frontend/
git commit -m "feat: scaffold vite react-ts frontend"
```

---

## Task 7: Configure Vite (proxy + Tailwind) and CSS

**Files:**
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Replace `frontend/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 2: Replace `frontend/src/index.css` entirely**

```css
@import "tailwindcss";

* {
  box-sizing: border-box;
}
```

- [ ] **Step 3: Remove default Tailwind imports if present in `main.tsx`**

Open `frontend/src/main.tsx`. Make sure it already imports `'./index.css'` (the Vite template does this). No other change needed.

- [ ] **Step 4: Verify CSS is applied**

```bash
cd frontend
npm run dev &
sleep 3
# Open http://localhost:5173 in a browser and confirm unstyled React page loads without build errors.
kill %1
```

- [ ] **Step 5: Commit**

```bash
cd ..
git add frontend/vite.config.ts frontend/src/index.css
git commit -m "feat: configure vite proxy and tailwind v4"
```

---

## Task 8: TypeScript types and API client

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Create `frontend/src/types.ts`**

```ts
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
```

- [ ] **Step 2: Create `frontend/src/api/` directory and `client.ts`**

```bash
mkdir -p frontend/src/api
```

`frontend/src/api/client.ts`:

```ts
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
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types.ts frontend/src/api/client.ts
git commit -m "feat: add frontend types and api client"
```

---

## Task 9: Chart component

**Files:**
- Create: `frontend/src/components/Chart.tsx`

- [ ] **Step 1: Create the component**

```bash
mkdir -p frontend/src/components
```

`frontend/src/components/Chart.tsx`:

```tsx
import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  ColorType,
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

    const series = chart.addCandlestickSeries({
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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/Chart.tsx
git commit -m "feat: add candlestick Chart component (lightweight-charts)"
```

---

## Task 10: SymbolSelector and JobStatus components

**Files:**
- Create: `frontend/src/components/SymbolSelector.tsx`
- Create: `frontend/src/components/JobStatus.tsx`

- [ ] **Step 1: Create `SymbolSelector.tsx`**

`frontend/src/components/SymbolSelector.tsx`:

```tsx
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
```

- [ ] **Step 2: Create `JobStatus.tsx`**

`frontend/src/components/JobStatus.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react'
import { fetchJobSummary, fetchJobs } from '../api/client'
import type { Job, JobSummary } from '../types'

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-yellow-400',
  running: 'text-blue-400',
  done: 'text-green-400',
  failed: 'text-red-400',
}

const STATUSES = ['pending', 'running', 'done', 'failed']
const PAGE_SIZE = 50

export function JobStatus() {
  const [summary, setSummary] = useState<JobSummary>({})
  const [jobs, setJobs] = useState<Job[]>([])
  const [total, setTotal] = useState(0)
  const [filter, setFilter] = useState('')
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    fetchJobSummary().then(setSummary).catch(console.error)
  }, [])

  const loadJobs = useCallback(() => {
    fetchJobs(filter || undefined, PAGE_SIZE, offset)
      .then((r) => {
        setJobs(r.jobs)
        setTotal(r.total)
      })
      .catch(console.error)
  }, [filter, offset])

  useEffect(() => {
    loadJobs()
  }, [loadJobs])

  const changeFilter = (s: string) => {
    setFilter(s)
    setOffset(0)
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {STATUSES.map((s) => (
          <div
            key={s}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800"
          >
            <div className="text-xs text-gray-500 uppercase tracking-wide">{s}</div>
            <div className={`text-3xl font-bold mt-1 ${STATUS_COLORS[s]}`}>
              {summary[s as keyof JobSummary] ?? 0}
            </div>
          </div>
        ))}
      </div>

      {/* Jobs table */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        {/* Filters */}
        <div className="px-4 py-3 border-b border-gray-800 flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500">Status:</span>
          {['', ...STATUSES].map((s) => (
            <button
              key={s || 'all'}
              onClick={() => changeFilter(s)}
              className={`text-xs px-2.5 py-1 rounded-full transition-colors ${
                filter === s
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {s || 'All'}
            </button>
          ))}
          <span className="ml-auto text-xs text-gray-500">{total.toLocaleString()} jobs</span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                <th className="px-4 py-2 font-medium">ID</th>
                <th className="px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Interval</th>
                <th className="px-4 py-2 font-medium">Month</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Completed</th>
                <th className="px-4 py-2 font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-600 text-xs">
                    No jobs found.
                  </td>
                </tr>
              )}
              {jobs.map((j) => (
                <tr
                  key={j.id}
                  className="border-b border-gray-800/60 hover:bg-gray-800/40 transition-colors"
                >
                  <td className="px-4 py-2 text-gray-500">{j.id}</td>
                  <td className="px-4 py-2">{j.symbol}</td>
                  <td className="px-4 py-2">{j.interval}</td>
                  <td className="px-4 py-2">
                    {j.year}-{String(j.month).padStart(2, '0')}
                  </td>
                  <td className={`px-4 py-2 font-medium ${STATUS_COLORS[j.status] ?? ''}`}>
                    {j.status}
                  </td>
                  <td className="px-4 py-2 text-gray-500 text-xs">
                    {j.completed_at
                      ? new Date(j.completed_at).toLocaleString()
                      : '—'}
                  </td>
                  <td
                    className="px-4 py-2 text-red-400/80 text-xs max-w-xs truncate"
                    title={j.error ?? ''}
                  >
                    {j.error ? j.error.split('\n')[0] : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="px-4 py-3 flex items-center justify-end gap-3 border-t border-gray-800">
          <button
            disabled={offset === 0}
            onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
            className="text-xs px-3 py-1 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            ← Previous
          </button>
          <span className="text-xs text-gray-500">
            {total === 0 ? '0' : `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)}`} of {total.toLocaleString()}
          </span>
          <button
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset((o) => o + PAGE_SIZE)}
            className="text-xs px-3 py-1 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: add SymbolSelector and JobStatus components"
```

---

## Task 11: App layout

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1: Replace `frontend/src/App.tsx`**

```tsx
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
```

- [ ] **Step 2: Verify `frontend/src/main.tsx` imports CSS**

Open `frontend/src/main.tsx`. Confirm it contains `import './index.css'`. If not, add it as the first line after the React imports. The default Vite template includes it — no change needed in most cases.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx frontend/src/main.tsx
git commit -m "feat: add app layout with chart and ingestion tabs"
```

---

## Task 12: Root dev script

**Files:**
- Modify: `package.json` (repo root)

- [ ] **Step 1: Install `concurrently` at the repo root**

```bash
# Run from tradan/ (repo root)
pnpm add -D concurrently
```

- [ ] **Step 2: Add dev scripts to root `package.json`**

Open `package.json` at the repo root and add the following inside the `"scripts"` object (alongside the existing bot scripts):

```json
"dev:backend": "cd backend && uv run uvicorn ingester.api:app --reload --port 8000",
"dev:frontend": "cd frontend && npm run dev",
"dev": "concurrently -n backend,frontend -c cyan,magenta \"pnpm dev:backend\" \"pnpm dev:frontend\""
```

The full `scripts` section should look like:

```json
"scripts": {
  "bot": "ts-node bot/strategy.ts",
  "init-strategy": "ts-node scripts/init-strategy.ts",
  "fund-strategy": "ts-node scripts/fund-strategy.ts",
  "build": "tsc",
  "typecheck": "tsc --noEmit",
  "dev:backend": "cd backend && uv run uvicorn ingester.api:app --reload --port 8000",
  "dev:frontend": "cd frontend && npm run dev",
  "dev": "concurrently -n backend,frontend -c cyan,magenta \"pnpm dev:backend\" \"pnpm dev:frontend\""
}
```

- [ ] **Step 3: Smoke test — start both services**

```bash
# From tradan/ repo root
pnpm dev &
sleep 5

# Backend health
curl -s http://localhost:8000/api/health

# Frontend dev server is live
curl -s http://localhost:5173 | grep -o '<title>.*</title>'

kill %1
```

Expected:
- `{"status":"ok"}` from backend
- `<title>Vite + React + TS</title>` (or similar) from frontend

- [ ] **Step 4: Test proxy (frontend → backend)**

With `pnpm dev` running, open `http://localhost:5173` in a browser.
- The **Chart** tab should load symbols and render a candlestick chart.
- The **Ingestion** tab should show job counts and a paginated table.

- [ ] **Step 5: Commit**

```bash
git add package.json pnpm-lock.yaml
git commit -m "feat: add root pnpm dev script with concurrently"
```

---

## Self-Review Checklist

- [x] `GET /api/health` — Task 3
- [x] `GET /api/symbols` — Task 4
- [x] `GET /api/klines` with optional `from_time` / `to_time` / `limit` — Task 4
- [x] `GET /api/jobs/summary` — Task 5
- [x] `GET /api/jobs` with status/symbol filter + pagination — Task 5
- [x] CORS configured for `http://localhost:5173` — Task 3
- [x] Vite proxy `/api` → `http://localhost:8000` — Task 7
- [x] Tailwind v4 via `@tailwindcss/vite` — Task 7
- [x] Candlestick chart with symbol/interval selector — Tasks 9, 10, 11
- [x] Job summary cards + paginated table + status filter — Task 10, 11
- [x] Root `pnpm dev` starts both services — Task 12
- [x] `get_conn` context manager used in all router handlers — Tasks 2, 4, 5
- [x] All API `time` values are open_time in **milliseconds**; Chart divides by 1000 before passing to lightweight-charts — Task 9
