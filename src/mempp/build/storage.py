from __future__ import annotations

from typing import Any

import numpy as np

from .types import ProceduralMemory


class _IndexStats:
    """Shape-compatible object for `index.describe_index_stats()` results.

    Provides an attribute `namespaces` (dict) mimicking Pinecone SDK's structure.
    """

    def __init__(self, namespaces: dict[str, dict[str, int]]):
        self.namespaces = namespaces


class PineconeMemoryStorage:
    """In-memory Pinecone-like storage used for local/dev and tests.

    This mimics a subset of the Pinecone client API and adds convenience state for
    retrieving full memory objects.
    """

    def __init__(
        self,
        api_key: str | None = None,
        environment: str | None = None,
        index_name: str = "mempp-memories",
        dimension: int = 1024,
        metric: str = "dotproduct",
        use_serverless: bool = True,
    ) -> None:
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.use_serverless = use_serverless

        # Public for other modules (retrieve/update) to touch
        self.memories: dict[str, ProceduralMemory] = {}
        self._vectors: dict[str, dict[str, np.ndarray]] = {  # namespace -> {id: vector}
            "proceduralized": {},
            "script": {},
            "trajectory": {},
        }
        self._metadata: dict[str, dict[str, dict[str, Any]]] = {  # namespace -> id -> metadata
            "proceduralized": {},
            "script": {},
            "trajectory": {},
        }

        # Minimal object that provides describe_index_stats()
        class _Index:
            def __init__(self, outer: PineconeMemoryStorage) -> None:
                self._outer = outer

            def describe_index_stats(self) -> _IndexStats:
                namespaces: dict[str, dict[str, int]] = {}
                for ns, table in self._outer._vectors.items():
                    namespaces[ns] = {"vector_count": len(table)}
                return _IndexStats(namespaces)

        self.index = _Index(self)

    # ===== CRUD =====

    async def store(self, memory: ProceduralMemory, namespace: str = "proceduralized") -> str:
        """Persist a memory and its vector/metadata to a namespace."""
        self.memories[memory.memory_id] = memory
        self._vectors.setdefault(namespace, {})[memory.memory_id] = memory.embedding.astype(np.float32)
        self._metadata.setdefault(namespace, {})[memory.memory_id] = {
            "usage_count": memory.usage_count,
            "success_rate": memory.success_rate,
            **memory.metadata,
        }
        return memory.memory_id

    def delete(self, memory_id: str, namespace: str = "proceduralized") -> None:
        """Remove a memory and its data from a namespace."""
        self._vectors.get(namespace, {}).pop(memory_id, None)
        self._metadata.get(namespace, {}).pop(memory_id, None)
        self.memories.pop(memory_id, None)

    def update_metadata(self, memory_id: str, metadata: dict[str, Any], namespace: str = "proceduralized") -> None:
        """Merge `metadata` into existing metadata for a memory in `namespace`."""
        ns_table = self._metadata.setdefault(namespace, {})
        base = ns_table.get(memory_id, {})
        base.update(metadata)
        ns_table[memory_id] = base
        if memory_id in self.memories:
            self.memories[memory_id].metadata.update(metadata)

    # ===== Query =====

    def _score_dense(self, q: np.ndarray, v: np.ndarray) -> float:
        if self.metric == "dotproduct":
            return float(np.dot(q, v))
        # fallback to cosine
        qn = q / (np.linalg.norm(q) + 1e-12)
        vn = v / (np.linalg.norm(v) + 1e-12)
        return float(np.dot(qn, vn))

    def _score_sparse(self, sparse_q: dict[str, float] | None, sparse_v: dict[str, float] | None) -> float:
        if not sparse_q or not sparse_v:
            return 0.0
        s = 0.0
        for k, w in sparse_q.items():
            if k in sparse_v:
                s += w * float(sparse_v[k])
        return float(s)

    def query(
        self,
        query_vector: np.ndarray,
        namespace: str,
        top_k: int,
        sparse_vector: dict[str, float] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top‑k matches from `namespace` with optional sparse and metadata filter."""
        # Collect candidates
        vecs = self._vectors.get(namespace, {})
        candidates: list[tuple[str, float]] = []
        for mid, vec in vecs.items():
            dense = self._score_dense(query_vector, vec)

            # Optional sparse component from memory object
            mem = self.memories.get(mid)
            sparse = self._score_sparse(sparse_vector, getattr(mem, "sparse_embedding", None)) if mem else 0.0

            score = 0.7 * dense + 0.3 * sparse if sparse_vector is not None else dense

            # Optional metadata filter (simple exact/threshold filters)
            if filter and mem is not None:
                ok = True
                for fk, cond in filter.items():
                    mv = mem.metadata.get(fk)
                    if isinstance(cond, dict):
                        # minimal: support {"$gte": x} and {"$lte": y}
                        gte = cond.get("$gte")
                        lte = cond.get("$lte")
                        if gte is not None and not (mv is not None and mv >= gte):
                            ok = False
                            break
                        if lte is not None and not (mv is not None and mv <= lte):
                            ok = False
                            break
                    else:
                        if mv != cond:
                            ok = False
                            break
                if not ok:
                    continue

            candidates.append((mid, score))

        # Top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[: max(0, top_k)]

        results: list[dict[str, Any]] = []
        for mid, sc in top:
            results.append({
                "id": mid,
                "score": float(sc),
                "metadata": self._metadata.get(namespace, {}).get(mid, {}),
            })
        return results

    # ===== Stats =====

    def get_statistics(self) -> dict[str, Any]:
        """Return total vector counts and per‑namespace counts."""
        namespaces: dict[str, dict[str, int]] = {}
        total = 0
        for ns, table in self._vectors.items():
            c = len(table)
            namespaces[ns] = {"vector_count": c}
            total += c
        return {"total_vectors": total, "namespaces": namespaces}
