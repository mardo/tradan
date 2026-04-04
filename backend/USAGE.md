# Binance Ingester — Usage Guide

Downloads Binance USDT-M futures monthly kline zips from
[data.binance.vision](https://data.binance.vision/?prefix=data/futures/um/monthly/klines/)
and stores them in a PostgreSQL database (standard Postgres or Neon).

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- A PostgreSQL database (local, Docker, or [Neon](https://neon.com))

---

## Setup

### 1. Install dependencies

From the `ingester/` directory:

```bash
uv sync
```

### 2. Configure the database connection

Copy the example env file and fill in your connection string:

```bash
cp .env.example .env
```

Edit `.env` in **`ingester/`** (next to `pyproject.toml`), not the monorepo root — that is the file the app loads.

```dotenv
# Standard Postgres
DATABASE_URL=postgresql://user:pass@localhost:5432/tradan

# Neon (use the direct or pooler hostname)
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
```

### 3. Run migrations

Creates the `klines` and `ingest_jobs` tables:

```bash
uv run ingest migrate
```

---

## Basic Workflow

### Step 1 — Enqueue jobs

Populate the job queue with the months you want to import:

```bash
# All intervals, all months from Jan 2020 to Mar 2026
uv run ingest enqueue --symbol BTCUSDT --start 2020-01 --end 2026-03

# Specific intervals only
uv run ingest enqueue --symbol BTCUSDT --interval 1m 1h --start 2024-01 --end 2026-03
```

`enqueue` uses `ON CONFLICT DO NOTHING`, so running it multiple times is safe — already-queued months are skipped.

### Step 2 — Run workers

```bash
# Single worker
uv run ingest run

# 4 parallel workers
uv run ingest run --workers 4
```

Workers claim jobs with `SELECT FOR UPDATE SKIP LOCKED`, so they never duplicate each other's work.

### Step 3 — Check progress

```bash
uv run ingest status
```

Example output:

```
Status      Count
----------------------
done          72
failed         1
pending       10
```

---

## Resume After Interruption

**Stopped with Ctrl-C or a clean quit** — pending jobs are untouched. Just run again:

```bash
uv run ingest run --workers 4
```

**Process was killed mid-job** — `ingest run` automatically detects any jobs left in `running` status from a previous crash and resets them to `pending` before workers start. No manual action needed.

**Failed jobs** — jobs that errored are marked `failed` and are not retried automatically. To re-queue them:

```bash
uv run ingest reset --failed
uv run ingest run --workers 4
```

---

## Reset

| Command | Effect |
|---|---|
| `uv run ingest reset` | Truncates `klines`, resets all jobs to `pending`. Re-run without re-enqueuing. |
| `uv run ingest reset --failed` | Re-queues only `failed` jobs. `done` rows are untouched. |
| `uv run ingest reset --hard` | Truncates `klines` **and** deletes all jobs. Full clean slate — you must re-enqueue. |

---

## All Commands

```
uv run ingest migrate
uv run ingest enqueue --symbol BTCUSDT --interval 1m --start 2020-01 --end 2026-03
uv run ingest run --workers 4
uv run ingest status
uv run ingest reset
uv run ingest reset --failed
uv run ingest reset --hard
```

Pass `--help` to any sub-command for full argument details:

```bash
uv run ingest enqueue --help
```

---

## Diagnostic logging (share with support / debugging)

All structured logs go to **stderr** with timestamps, process id, and (for worker lines) `job_id` plus a `symbol/interval YYYY-MM` label.

```bash
# Verbose: log every download URL, batch inserts, HTTP body previews on errors
PYTHONUNBUFFERED=1 uv run ingest run --workers 4 --log-level DEBUG 2>&1 | tee ingest-debug.log
```

Or set `INGEST_LOG_LEVEL=DEBUG` in `.env` (used when you omit `--log-level`).

What to look for in a log excerpt:

- **`ingester.downloader`**: HTTP status, 404 vs other errors, optional response body preview.
- **`ingester.worker`**: `Claimed job`, `Start job url=…`, `Downloaded zip bytes=…`, `Job done`, or `Job failed` with stack trace.

---

## Adding Future Migrations

Place a new numbered SQL file in `ingester/migrations/`, e.g.:

```
migrations/
├── 001_initial.sql
└── 002_add_index.sql
```

Then run:

```bash
uv run ingest migrate
```

Migration files are applied in alphabetical order. Already-applied files are tracked in the `_migrations` table and skipped automatically.
