from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .storage import PineconeMemoryStorage
from .types import (
    EmbeddingModel,
    MultilingualE5Embedder,
    ProceduralizedMemory,
    ProceduralMemory,
    ScriptMemory,
    TaskStatus,
    Trajectory,
    TrajectoryMemory,
)


@dataclass
class MemppBuildPipeline:
    """Minimal memory build pipeline compatible with retrieve/update/system.

    This implementation focuses on shape compatibility and local development.
    It does NOT call external LLMs; instead it constructs memories deterministically
    from the provided `Trajectory` and generates an embedding via the provided
    `EmbeddingModel` (default: `MultilingualE5Embedder`).
    """

    pinecone_api_key: str | None = None
    storage: PineconeMemoryStorage | None = None
    embedder: EmbeddingModel = field(default_factory=MultilingualE5Embedder)
    llm_client: Any | None = None

    async def build_from_trajectory(
        self, trajectory: Trajectory, strategy: str = "proceduralization"
    ) -> ProceduralMemory:
        """Build a procedural memory from a trajectory using a simple heuristic pipeline.

        - strategy="trajectory": stores the raw trajectory details
        - strategy="script": extracts step-like strings from actions/states
        - strategy="proceduralization" (default): combines both into a compact record
        """
        # Ensure embedder is initialized for type-checkers
        assert self.embedder is not None
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

        match strategy:
            case "trajectory":
                memory: ProceduralMemory = TrajectoryMemory(
                    memory_id=mem_id,
                    task_pattern=task_pattern,
                    trajectory=trajectory,
                    key_states=[s.content for s in trajectory.states[:5]],
                    critical_actions=[a.content for a in trajectory.actions[:5]],
                    embedding=emb,
                    sparse_embedding=sparse,
                )

            case "script":
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

            case _:  # default: proceduralization (combine)
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

    # Optional stats helper to satisfy callers that expect it
    def get_build_statistics(self) -> dict[str, Any]:
        """Return lightweight build stats for local/dev usage."""
        return {"total_processed": 0, "successful_builds": 0, "failed_builds": 0, "average_build_time": 0.0}

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


__all__ = ["MemppBuildPipeline"]
