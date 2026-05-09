"""Load a saved SB3 model and run inference."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO, SAC


_ALGO_REGISTRY = {"A2C": A2C, "PPO": PPO, "SAC": SAC}


@dataclass(frozen=True)
class InferenceResult:
    action: np.ndarray
    inference_ms: int


class ModelRunner:
    def __init__(self, model_path: Path, algorithm: str):
        cls = _ALGO_REGISTRY.get(algorithm.upper())
        if cls is None:
            raise ValueError(f"unknown algorithm: {algorithm}")
        self._model = cls.load(str(model_path))

    def predict(self, obs: dict) -> InferenceResult:
        t0 = time.perf_counter_ns()
        action, _ = self._model.predict(obs, deterministic=True)
        elapsed_ns = time.perf_counter_ns() - t0
        return InferenceResult(
            action=np.asarray(action, dtype=np.float32),
            inference_ms=int(elapsed_ns // 1_000_000),
        )
