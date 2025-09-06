from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .storage import PineconeMemoryStorage
from .types import (
    EmbeddingModel,
    MultilingualE5Embedder,
    ProceduralMemory,
    ProceduralizedMemory,
    ScriptMemory,
    TaskStatus,
    Trajectory,
    TrajectoryMemory,
)


@dataclass
class MempBuildPipeline:
    """Minimal memory build pipeline compatible with retrieve/update/system.

    This implementation focuses on shape compatibility and local development.
    It does NOT call external LLMs; instead it constructs memories deterministically
    from the provided `Trajectory` and generates an embedding via the provided
    `EmbeddingModel` (default: `MultilingualE5Embedder`).
    """

    pinecone_api_key: Optional[str] = None
    storage: Optional[PineconeMemoryStorage] = None
    embedder: Optional[EmbeddingModel] = None
    llm_client: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = MultilingualE5Embedder()

    async def build_from_trajectory(
        self, trajectory: Trajectory, strategy: str = "proceduralization"
    ) -> ProceduralMemory:
        # Derive a task pattern from task description (or states/actions fallback)
        task_pattern = trajectory.task_description or (trajectory.states[0].content if trajectory.states else "")

        # Produce a single input text for embedding
        summary = self._summarize(trajectory)
        emb = self.embedder.encode(summary)[0]

        # Rudimentary sparse embedding: token counts
        sparse: dict[str, float] = {}
        for tok in summary.lower().split():
            sparse[tok] = sparse.get(tok, 0.0) + 1.0

        mem_id = str(uuid.uuid4())

        if strategy == "trajectory":
            memory = TrajectoryMemory(
                memory_id=mem_id,
                task_pattern=task_pattern,
                trajectory=trajectory,
                key_states=[s.content for s in trajectory.states[:5]],
                critical_actions=[a.content for a in trajectory.actions[:5]],
                embedding=emb,
                sparse_embedding=sparse,
            )
        elif strategy == "script":
            steps = self._heuristic_steps(trajectory)
            memory = ScriptMemory(
                memory_id=mem_id,
                task_pattern=task_pattern,
                script="\n".join(steps),
                steps=steps,
                preconditions=[],
                postconditions=[],
                expected_outcomes={"status": trajectory.status.name},
                embedding=emb,
                sparse_embedding=sparse,
            )
        else:  # default: proceduralization (combine)
            steps = self._heuristic_steps(trajectory)
            memory = ProceduralizedMemory(
                memory_id=mem_id,
                task_pattern=task_pattern,
                trajectory=trajectory,
                script="\n".join(steps),
                abstraction_level=0.5,
                key_patterns=steps[:3],
                embedding=emb,
                sparse_embedding=sparse,
            )

        # Basic initialization of success/usage signals
        memory.success_rate = (
            1.0
            if trajectory.status == TaskStatus.SUCCESS
            else (0.5 if trajectory.status == TaskStatus.PARTIAL else 0.0)
        )
        memory.usage_count = 0
        memory.metadata.update({
            "built_from": "trajectory",
            "strategy": strategy,
        })

        # Optional: store immediately if a storage is provided
        if self.storage is not None:
            await self.storage.store(memory, namespace=self._namespace_for(memory))

        return memory

    def _namespace_for(self, memory: ProceduralMemory) -> str:
        if isinstance(memory, TrajectoryMemory):
            return "trajectory"
        if isinstance(memory, ScriptMemory):
            return "script"
        return "proceduralized"

    def _summarize(self, tr: Trajectory) -> str:
        parts: list[str] = [tr.task_description]
        if tr.actions:
            parts.extend([a.content for a in tr.actions[:5]])
        if tr.observations:
            parts.extend([o.content for o in tr.observations[:3]])
        return " \n ".join([p for p in parts if p])

    def _heuristic_steps(self, tr: Trajectory) -> list[str]:
        # Turn actions or states into a step list; fall back to splitting description
        if tr.actions:
            return [f"- {a.content}" for a in tr.actions]
        if tr.states:
            return [f"- {s.content}" for s in tr.states]
        return [f"- {s.strip()}" for s in tr.task_description.split(";") if s.strip()]


__all__ = ["MempBuildPipeline"]
