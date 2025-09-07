from __future__ import annotations

from datetime import datetime

import numpy as np

from mempp.build import (
    Action,
    ActionType,
    MemppBuildPipeline,
    Observation,
    PineconeMemoryStorage,
    State,
    TaskStatus,
    Trajectory,
)


class ToyEmbedder:
    def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=1024).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.stack(vecs)


async def test_build_three_strategies() -> None:
    storage = PineconeMemoryStorage(index_name="mempp-test", dimension=1024)
    pipeline = MemppBuildPipeline(storage=storage, embedder=ToyEmbedder())

    traj = Trajectory(
        task_id="t-2",
        task_description="Organize kitchen utensils on the shelf",
        states=[State("start", datetime.now())],
        actions=[Action(ActionType.TOOL_USE, "organize", datetime.now())],
        observations=[Observation("done", datetime.now(), 0.5)],
        status=TaskStatus.PARTIAL,
        final_reward=0.5,
    )

    mem_proc = await pipeline.build_from_trajectory(traj, strategy="proceduralization")
    mem_traj = await pipeline.build_from_trajectory(traj, strategy="trajectory")
    mem_script = await pipeline.build_from_trajectory(traj, strategy="script")

    assert mem_proc.memory_id != mem_traj.memory_id != mem_script.memory_id
    assert storage.get_statistics()["total_vectors"] >= 3
