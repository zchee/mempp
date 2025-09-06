from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np

# ===== Task/Trajectory primitives =====


class TaskStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    PARTIAL = auto()


class ActionType(Enum):
    PLANNING = auto()
    TOOL_USE = auto()
    OBSERVATION = auto()


@dataclass
class State:
    content: str
    timestamp: datetime


@dataclass
class Action:
    action_type: ActionType
    content: str
    timestamp: datetime


@dataclass
class Observation:
    content: str
    timestamp: datetime
    reward: float = 0.0


@dataclass
class Trajectory:
    task_id: str
    task_description: str
    states: list[State]
    actions: list[Action]
    observations: list[Observation]
    status: TaskStatus
    final_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ===== Memory objects =====


@dataclass
class ProceduralMemory:
    memory_id: str
    task_pattern: str
    embedding: np.ndarray
    sparse_embedding: dict[str, float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def increment_usage(self, success: bool) -> None:
        self.usage_count += 1
        # Exponential moving average for success rate
        alpha = 0.2
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)


@dataclass
class TrajectoryMemory(ProceduralMemory):
    trajectory: Trajectory = field(default_factory=lambda: Trajectory("", "", [], [], [], TaskStatus.PARTIAL, 0.0))
    key_states: list[str] = field(default_factory=list)
    critical_actions: list[str] = field(default_factory=list)


@dataclass
class ScriptMemory(ProceduralMemory):
    script: str = ""
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    expected_outcomes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralizedMemory(ProceduralMemory):
    trajectory: Trajectory | None = None
    script: str = ""
    abstraction_level: float = 0.0
    key_patterns: list[str] = field(default_factory=list)


# ===== Embedding interfaces =====


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str] | str) -> np.ndarray:  # (n, d)
        ...


class MultilingualE5Embedder:
    """Lightweight, dependency‑free placeholder for a multilingual embedding model.

    - Deterministically maps text -> vector using SHA256 as a seed.
    - Output dimension defaults to 1024 to match system config.
    - Suitable for tests and local development without network/model downloads.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _hash_to_vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand deterministically to the target dimension
        rng_seed = int.from_bytes(h[:8], "big", signed=False) % (2**32)
        rng = np.random.default_rng(rng_seed)
        v = rng.normal(size=(self.dimension,)).astype(np.float32)
        # L2 normalize for cosine/dotproduct stability
        norm = np.linalg.norm(v) + 1e-12
        return (v / norm).astype(np.float32)

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        mat = np.stack([self._hash_to_vec(t) for t in texts], axis=0)
        return mat


__all__ = [
    "TaskStatus",
    "ActionType",
    "State",
    "Action",
    "Observation",
    "Trajectory",
    "ProceduralMemory",
    "TrajectoryMemory",
    "ScriptMemory",
    "ProceduralizedMemory",
    "EmbeddingModel",
    "MultilingualE5Embedder",
]
