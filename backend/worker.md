# Training worker (sequential, same host)

Run a **single** process that claims pending `model_configs` from Postgres, trains one run at a time, then repeats until the queue is empty. Same machine as the rest of `backend/`: use one `DATABASE_URL`, leave **`MODELS_RSYNC_TARGET` unset** (artifacts stay in `trained_models/`; no rsync).

---

## TL;DR

```bash
cd backend && uv sync
# DATABASE_URL in backend/.env
uv run ingest migrate

# Phase 1 — register 63 configs (names include _p1_; required for sweep_phase2.py)
uv run python scripts/sweep_phase1.py
uv run train worker
uv run train winners

# Phase 2 — hyperparam grid from top Phase 1 winners (evaluated); see scripts/sweep_phase2.py
uv run python scripts/sweep_phase2.py
uv run train worker

# Phase 3 — long runs from top Phase 2 winners; see scripts/sweep_phase3.py
uv run python scripts/sweep_phase3.py
uv run train worker

# Phase 4 (optional) — same winners, higher total_timesteps (e.g. 5–10M); register then:
uv run train worker
```

---

## Phases 1–3 (and 4)

Rough plan from [TRAINING_AT_SCALE.md](TRAINING_AT_SCALE.md) — each phase narrows the grid using results from the previous one.

| Phase | Goal | What you vary | ~Runs |
|------|------|----------------|-------|
| **1** | Find strong **interval × algorithm** pairs | BTC only, single symbol, all columns, lookback 500, LR 3e-4; **7 intervals × 3 algos × 3 seeds** | ~63 |
| **2** | Generalize winners across symbols | Top **3** interval+algo combos from Phase 1 × **5 targets** × **2** symbol sets (single vs multi+BTC) × **3 seeds** | ~90 |
| **3** | Tune hyperparameters | Top **5** configs from Phase 2 × lookbacks, LRs, column subsets × seeds (see scale doc for full grid) | ~150+ |
| **4** | Long training | Top configs from Phase 3, **5–10M** timesteps instead of ~1M | small batch |

Use `uv run train winners` and SQL under `infra/scripts/` (`winners.sql`, etc.) to rank runs between phases.

**Repo scripts:** Phase sweeps live in **`backend/scripts/`** (`sweep_phase1.py`, `sweep_phase2.py`, `sweep_phase3.py`). They only need `DATABASE_URL` in `backend/.env` and normal `uv run` imports — no `sys.path` hacks. Names use `_p1_`, `_p2_`, `_p3_` segments so Phase 2/3 queries can find prior winners.

The phase table below matches [TRAINING_AT_SCALE.md](TRAINING_AT_SCALE.md) conceptually; the **implemented** Phase 2/3 scripts follow the infra pipeline (Phase 2 = lookback × LR × seeds on top evaluated Phase 1 winners; Phase 3 = 5M-step reruns of top Phase 2 winners). For the broader “targets × symbol sets” Phase 2 from the doc, add another script or extend `sweep_phase2.py`.

---

## Phase 1: register configs

From `backend/`:

```bash
uv run python scripts/sweep_phase1.py
```

Ensure klines are ingested for **BTCUSDT** at every interval in the script. Re-running **updates** rows with the same `name` (`ON CONFLICT`).

**After Phase 1:** run holdout evaluation for models you care about (e.g. `infra/scripts/evaluate_winners.sh` on your stack), then `uv run python scripts/sweep_phase2.py` when you have evaluated Phase 1 winners in the DB.

---

## Prerequisites

- [TRAINER.md](TRAINER.md): Python 3.12+, `uv`, `DATABASE_URL` in **`backend/.env`**, ingested data for your symbols/intervals.
- Schema: `uv run ingest migrate`.

---

## Run the worker

```bash
cd backend
uv run train worker
```

- `--poll-seconds N` — when the queue is empty, sleep and recheck (default `0` = exit).
- `--cpu-usage PCT` — OpenMP/MKL thread cap (default `85`; see `train worker --help`).

---

## Operations notes

- **Stuck claims:** after a crash, use `uv run train release-claims --older-than-seconds …` or wait for claim staleness (see `claim_pending_model` in code).
- **Inspect:** `uv run train list`, `uv run train status --run <id>`, `uv run train winners`.
- **Evaluate:** `uv run train evaluate --model <name> --run <id>`.
- One-off training without the queue: `uv run train start --model <name>`.
