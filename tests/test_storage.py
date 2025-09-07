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


async def test_storage_build_and_query() -> None:
    storage = PineconeMemoryStorage(index_name="mempp-test", dimension=1024)
    pipeline = MemppBuildPipeline(storage=storage, embedder=ToyEmbedder())

    traj = Trajectory(
        task_id="t-1",
        task_description="Clean the cup and put it in the microwave",
        states=[State("cup dirty", datetime.now())],
        actions=[
            Action(ActionType.PLANNING, "plan cleaning", datetime.now()),
            Action(ActionType.TOOL_USE, "wash cup", datetime.now()),
            Action(ActionType.TOOL_USE, "place in microwave", datetime.now()),
        ],
        observations=[Observation("cup is clean", datetime.now(), 0.8)],
        status=TaskStatus.SUCCESS,
        final_reward=1.0,
    )

    mem = await pipeline.build_from_trajectory(traj)

    # Query directly via storage (dense only)
    q = ToyEmbedder().encode("put clean cup into microwave")[0]
    matches = storage.query(query_vector=q, namespace="proceduralized", top_k=3)

    assert matches, "expected at least one match"
    assert matches[0]["id"] == mem.memory_id
