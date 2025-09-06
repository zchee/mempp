from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import ProceduralMemory


@dataclass
class _NamespaceStats:
    vector_count: int = 0


class _IndexStats:
    """Shape-compatible object for `index.describe_index_stats()` results.

    Provides an attribute `namespaces` (dict) mimicking Pinecone SDK's structure
    that code in update/system expects.
    """

    def __init__(self, namespaces: Dict[str, Dict[str, int]]):
        self.namespaces = namespaces


class PineconeMemoryStorage:
    """In-memory Pinecone-like storage used for local/dev and tests.

    This mimics a subset of the Pinecone client API and adds convenience state for
    retrieving full memory objects.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str | None = None,
        index_name: str = "memp-memories",
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
        self.memories: Dict[str, ProceduralMemory] = {}
        self._vectors: Dict[str, Dict[str, np.ndarray]] = {  # namespace -> {id: vector}
            "proceduralized": {},
            "script": {},
            "trajectory": {},
        }
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {  # namespace -> id -> metadata
            "proceduralized": {},
            "script": {},
            "trajectory": {},
        }

        # Minimal object that provides describe_index_stats()
        class _Index:
            def __init__(inner_self, outer: "PineconeMemoryStorage") -> None:
                inner_self._outer = outer

            def describe_index_stats(inner_self) -> _IndexStats:
                namespaces: Dict[str, Dict[str, int]] = {}
                for ns, table in inner_self._outer._vectors.items():
                    namespaces[ns] = {"vector_count": len(table)}
                return _IndexStats(namespaces)

        self.index = _Index(self)

    # ===== CRUD =====

    async def store(self, memory: ProceduralMemory, namespace: str = "proceduralized") -> str:
        self.memories[memory.memory_id] = memory
        self._vectors.setdefault(namespace, {})[memory.memory_id] = memory.embedding.astype(np.float32)
        self._metadata.setdefault(namespace, {})[memory.memory_id] = {
            "usage_count": memory.usage_count,
            "success_rate": memory.success_rate,
            **memory.metadata,
        }
        return memory.memory_id

    def delete(self, memory_id: str, namespace: str = "proceduralized") -> None:
        self._vectors.get(namespace, {}).pop(memory_id, None)
        self._metadata.get(namespace, {}).pop(memory_id, None)
        self.memories.pop(memory_id, None)

    def update_metadata(self, memory_id: str, metadata: Dict[str, Any], namespace: str = "proceduralized") -> None:
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

    def _score_sparse(self, sparse_q: Optional[Dict[str, float]], sparse_v: Optional[Dict[str, float]]) -> float:
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
        sparse_vector: Optional[Dict[str, float]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Collect candidates
        vecs = self._vectors.get(namespace, {})
        candidates: List[Tuple[str, float]] = []
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

        results: List[Dict[str, Any]] = []
        for mid, sc in top:
            results.append({
                "id": mid,
                "score": float(sc),
                "metadata": self._metadata.get(namespace, {}).get(mid, {}),
            })
        return results

    # ===== Stats =====

    def get_statistics(self) -> Dict[str, Any]:
        namespaces: Dict[str, Dict[str, int]] = {}
        total = 0
        for ns, table in self._vectors.items():
            c = len(table)
            namespaces[ns] = {"vector_count": c}
            total += c
        return {"total_vectors": total, "namespaces": namespaces}
