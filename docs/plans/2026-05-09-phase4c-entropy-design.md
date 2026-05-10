# Phase 4C — A2C Entropy Regularization Design

**Date:** 2026-05-09
**Status:** Design (pre-implementation)
**Depends on:** Phase 4A (`docs/plans/2026-05-09-phase4a-seed-robustness.md`) — adds the `ModelConfig.seed` plumbing this experiment also relies on.

---

## Motivation

Phase 4A ran 5 seeded training runs each on the three top-3 P2 architectures (lb=100/250/500, A2C, lr=3e-4, 1M timesteps). All three architectures **failed** the seed-robustness gate (median holdout PnL > 0 AND ≥3 of 5 seeds positive):

| arch  | seeds positive | median holdout PnL | decision |
|-------|---------------:|-------------------:|----------|
| lb100 | 0 / 5          | -$2,378            | FAIL     |
| lb250 | 1 / 5          | -$10,510           | FAIL     |
| lb500 | 0 / 5          | -$10,155           | FAIL     |

Two distinct failure modes are visible in the per-seed detail:
- **Do-nothing collapse** (lb100 dominant). 4 of 5 lb100 training runs ended at exactly $0 with 0 trades — the policy learned to never act. Classic A2C entropy decay: as training proceeds, the policy distribution sharpens to a degenerate "no action" delta.
- **Active-but-losing** (lb500 dominant). lb500 produced trades (10–117 per holdout) but lost on every seed, often blowing up the account at >100% drawdown. The policy *did* learn a directional bet but overfit a transient training pattern.

Phase 4C tests the leading hypothesis for the first failure mode: **entropy decay**. Stable-baselines3 A2C exposes an `ent_coef` hyperparameter (default 0.0). Adding a small positive entropy bonus to the loss prevents the policy from collapsing to a deterministic delta — it pays a cost proportional to `-H(π)`, which is large when π is uniform and small when π is concentrated. With `ent_coef=0.01`, the policy retains exploration pressure throughout training.

If entropy reg rescues the architecture, the seed-robustness gate becomes attainable and 4A should be re-run with the new training config across all three lookbacks. If it doesn't, both algorithmic levers (seeds, entropy) have been tested and the next gate becomes the env/reward audit (master plan's Path C).

## Non-goals (YAGNI)

- Sweeping multiple `ent_coef` values. The plan picks one (0.01) and tests it. A sweep can come after if 0.01 is in the wrong order of magnitude.
- Extending to lb100 and lb250. The 4A failure modes for those arches included pure do-nothing (lb100) and high-variance (lb250); the cleanest test of the entropy hypothesis is lb500, which produced *active* policies that nonetheless failed. If 4C lb500 passes, we then run a follow-up across lb100/lb250 (as a re-run of 4A with the new arch).
- PPO. The master plan's 4C section also proposed a PPO sweep at lb500. That's deliberately out of scope here — testing one variable at a time. PPO is a candidate follow-up if A2C+entropy fails.
- Walk-forward eval. That's Phase 4B; out of scope here.

## Experiment specification

| field | value |
|---|---|
| symbol / interval / algorithm | BTCUSDT / 4h / A2C |
| lookback_window | 500 |
| learning_rate | 3e-4 |
| **ent_coef** | **0.01** |
| total_timesteps | 2,000,000 |
| seeds | `[1001, 2002, 3003, 4004, 5005]` (paired with 4A; same seed values across the two experiments) |
| number of runs | 5 |
| pass criterion | median holdout PnL > 0 AND ≥ 3 of 5 seeds positive (same as 4A) |
| comparison baseline | the existing five `_lb500_3em4_p4a_s{0..4}` runs from 4A (1M timesteps, ent_coef=0) |

Seed-pairing rationale: by reusing the exact seed values used in 4A, we get a paired comparison per seed (e.g., 4A_s0 vs 4C_s0 both use seed 1001). If entropy reg helps, the paired difference should be systematically positive.

## Architecture

Two code changes plus two new operational scripts. Mirrors the 4A pattern exactly so the diff is small and the operational flow is familiar.

### `ModelConfig.ent_coef` field

`backend/src/trainer/config.py` gains:

```python
    # Coefficient on the entropy bonus added to the policy-gradient loss. SB3
    # default is 0.0 for A2C/PPO and 'auto' (a learnable schedule) for SAC.
    # Set positive to keep the policy from collapsing to a deterministic delta —
    # see Phase 4C study (docs/plans/2026-05-09-phase4c-entropy-design.md).
    ent_coef: float = 0.0
```

`to_dict` adds `"ent_coef": self.ent_coef` next to `learning_rate`/`seed`.
`from_dict` already filters by `__dataclass_fields__`, no change.

### Conditional pass-through in `train_model`

`backend/src/trainer/training/trainer.py` (around lines 294–299) goes from

```python
model = algo_cls(
    "MultiInputPolicy",
    env,
    learning_rate=config.learning_rate,
    seed=config.seed,
    verbose=0,
)
```

to

```python
algo_kwargs: dict = {}
if config.ent_coef > 0:
    algo_kwargs["ent_coef"] = config.ent_coef

model = algo_cls(
    "MultiInputPolicy",
    env,
    learning_rate=config.learning_rate,
    seed=config.seed,
    verbose=0,
    **algo_kwargs,
)
```

Why conditional: SAC accepts `ent_coef` but its semantics differ — the default `'auto'` is a learnable schedule. Forcing it to `0.0` would silently disable SAC's entropy mechanism. Gating on `> 0` preserves the SB3 default for any algo when the field is unset, and only overrides when the user opts in. (For our actual experiment we always pass to A2C with `0.01`, so this conditional is just defensive for future SAC use.)

### `backend/scripts/sweep_phase4c_entropy.py`

Mirrors `sweep_phase4a.py`. Constants:

```python
LOOKBACK = 500
SEEDS = [1001, 2002, 3003, 4004, 5005]
LEARNING_RATE = 3e-4
ENT_COEF = 0.01
TIMESTEPS = 2_000_000
INTERVAL = "4h"
ALGO = "A2C"
PHASE = "p4c"
ENT_SLUG = "ent01"  # 0.01 → "ent01" — keeps the door open for ent005, ent05 follow-ups
```

`build_phase4c_entropy_configs() -> list[ModelConfig]` returns 5 configs named:
`btc_4h_a2c_lb500_3em4_ent01_p4c_s{0..4}`

Each carries `seed = SEEDS[i]`, `ent_coef = 0.01`, `total_timesteps = 2_000_000`. `main()` iterates and calls `save_model_config(cfg)`.

### `backend/scripts/phase4c_entropy_summary.py`

Reuses `phase4a_summary.evaluate_arch` (it's pure — same decision logic). Since `backend/scripts/` is not an importable package, the summary script loads its sibling via `importlib.util` (the same pattern the unit tests already use to load these scripts):

```python
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "phase4a_summary",
    Path(__file__).resolve().parent / "phase4a_summary.py",
)
_p4a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p4a)
evaluate_arch = _p4a.evaluate_arch
```

The alternative — moving `evaluate_arch` to `backend/src/trainer/` as a real module — is cleaner but is a refactor we can defer until a third phase summary needs it (YAGNI).

Queries the holdout-eval rows for `%_lb500_3em4_ent01_p4c_s%` and the matched 4A baseline `%_lb500_3em4_p4a_s%`. Prints:

```
arch        seeds  pos     median PnL     4A median  decision
--------------------------------------------------------------------
lb500_ent01     5    ?         $...          $-10,155  PASS|FAIL
```

Followed by a paired per-seed diff table:

```
seed   4A holdout PnL   4C holdout PnL   delta
1001            -$5,195            $...    $...
2002           -$10,155            $...    $...
...
```

The `delta` column is the headline diagnostic: if entropy reg systematically helps, deltas are positive across seeds. If it's a coin flip, the rescue claim is suspect even if median crosses zero.

### Tests

`backend/tests/trainer/test_seed_plumbing.py` already exists. Append:
- `test_model_config_ent_coef_default_is_zero` — round-trip default of 0.0.
- `test_model_config_ent_coef_round_trip` — set 0.01, round-trip via to_dict/from_dict.
- `test_train_model_passes_ent_coef_when_set` — source-inspection: assert `algo_kwargs["ent_coef"] = config.ent_coef` and the conditional gating is present in `train_model`.

`backend/tests/trainer/test_phase4c_entropy.py` (new):
- `test_phase4c_builder_produces_5_configs` — same shape as 4A's builder test, asserts 5 configs with correct names, seeds, ent_coef, timesteps.
- `test_phase4c_summary_reuses_evaluate_arch` — imports `evaluate_arch` from the 4A summary module and verifies it's the same function (no logic duplication).

`evaluate_arch` itself doesn't need re-testing — already covered by 4A's 5 unit tests.

## Operational sequence

Same pattern as 4A, on the training host (69.197.128.130, `/opt/tradan`):

1. **Stop watchdog** — `systemctl stop tradan-train-worker-health.service`. The watchdog has the same torch.set_num_interop_threads bug; use the per-process driver instead.
2. **Switch branch** — checkout the 4C feature branch on the training host so the new code is present.
3. **Register configs** — `cd /opt/tradan/backend && uv run python scripts/sweep_phase4c_entropy.py`.
4. **Sequential drive script** — `/tmp/p4c_drive.sh` (analogous to `/tmp/p4a_drive.sh`): loops through pending `_p4c_` configs and runs `train start --model <name>` for each, one at a time. ~30 min per run × 5 = ~2.5h wall-clock.
5. **Eval** — per-name eval loop (same fallback as 4A; `evaluate_winners.sh` would still be contaminated by stale non-p4c rows). ~3 min total.
6. **Decision** — `uv run python scripts/phase4c_entropy_summary.py`. Capture the table and the paired per-seed diff.
7. **Plan update** — append a `**4C Outcome:**` subsection to `docs/plans/2026-05-09-phase4-training-plan.md` with the decision matrix, paired diff, and verdict on the entropy-collapse hypothesis. Commit on the 4C feature branch.
8. **Wrap up** — push branch, open PR, restart watchdog with `systemctl start tradan-train-worker-health.service`.

Total wall-clock: ~3 h (registration <1 min, training ~2.5h, eval ~3 min, summary + writeup ~10 min).

## Decision tree

After the experiment lands and the writeup is in:

- **Pass (median > 0 AND ≥3/5 positive AND paired delta systematically positive)**
  → Entropy hypothesis confirmed for lb500. Next step: re-run 4A on lb100 and lb250 with `ent_coef=0.01`, 2M timesteps. If those also pass, **then** consider promotions.
- **Pass on median but paired delta is mixed** (e.g., 4C is better on 3 seeds but worse on 2)
  → Result is fragile. Continue to a small `ent_coef` sweep (0.005, 0.05) before declaring success.
- **Fail (still losing on holdout)**
  → Entropy reg didn't rescue lb500. Both algorithmic levers (seed plumbing, entropy reg) have now been tried and failed. Next gate: env/reward audit (Path C in master plan). Specifically inspect `TradingEnv` leverage caps, fee parameters, and reward function — drawdowns >100% on holdout strongly suggest the env permits account blow-ups that should be impossible under realistic trading constraints.

## Risks and tradeoffs

- **Confound between entropy and timesteps.** 4A used 1M timesteps; 4C uses 2M. If 4C passes, we can't fully isolate "entropy helped" vs "more training helped" — *but* Phase 3 already trained 5M timesteps without entropy and failed catastrophically, so additional timesteps alone is not a sufficient explanation. We treat 4C-pass as "entropy reg made longer training viable" rather than "more timesteps fixed it."
- **`ent_coef=0.01` may be wrong magnitude.** SB3's PPO default is 0.0 but 0.01 is the conventional first-try for A2C in continuous-action environments. If the result is borderline (e.g., 2/5 positive with median near zero), a small sweep is the next step rather than a verdict.
- **Holdout window unchanged.** 4A's holdout is the last ~16 months. If that period happens to be a regime no policy can profit in, we wouldn't see a "pass" even with a perfectly trained model. This is exactly the 4B walk-forward question; it's deferred but worth keeping in mind when interpreting a borderline 4C fail.
- **Paired-seed comparison assumes seeded runs are reasonably reproducible.** GPU non-determinism means same-seed reruns aren't bit-identical (documented in `ModelConfig.seed`'s field comment). For statistical paired-diff purposes the per-seed correlation should still be high enough to extract signal, but a 0.5-sigma noise floor around each delta is realistic.

## Out of scope

- Adding a config for `n_steps`, `gamma`, or other A2C hyperparams. If 4C shows promise, future experiments may want them, but YAGNI for this study.
- Generalizing the conditional pass-through to a `algo_kwargs: dict` field on `ModelConfig`. Single-field is enough for the next 1–2 experiments. We can refactor when a third hyperparam needs the same treatment.
- Modifying the watchdog to fix the torch interop bug. The bug is documented in the master plan; fixing it is its own change with its own review surface.
