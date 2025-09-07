from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class RetrievalStrategyName(Enum):
    QUERY_BASED = "QUERY_BASED"
    HYBRID = "HYBRID"
    NAMESPACE_AWARE = "NAMESPACE_AWARE"
    CASCADING = "CASCADING"


class UpdateStrategyName(Enum):
    VANILLA = "VANILLA"
    VALIDATION = "VALIDATION"
    ADJUSTMENT = "ADJUSTMENT"
    DEPRECATION = "DEPRECATION"
    CONSOLIDATION = "CONSOLIDATION"
    PRUNING = "PRUNING"
    REINFORCEMENT = "REINFORCEMENT"
    NAMESPACE_MIGRATION = "NAMESPACE_MIGRATION"


@dataclass
class MemppSystemConfig:
    """Lightweight system configuration used by the CLI and orchestrator.

    This module intentionally avoids importing heavy dependencies so that
    simple invocations (like `mempp info`) remain fast and side‑effect free.
    """

    # Pinecone configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "mempp-memories"
    pinecone_metric: str = "dotproduct"
    pinecone_use_serverless: bool = True
    pinecone_dimension: int = 1024

    # Namespace configuration
    default_namespaces: list[str] = field(default_factory=lambda: ["proceduralized", "script", "trajectory"])
    namespace_auto_balance: bool = True
    max_vectors_per_namespace: int = 5000

    # Build configuration
    build_strategy: str = "proceduralization"
    enable_sparse_vectors: bool = True

    # Retrieval configuration
    retrieval_strategy: RetrievalStrategyName = RetrievalStrategyName.CASCADING
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranking: bool = True
    use_caching: bool = True
    use_hybrid_search: bool = True
    alpha: float = 0.7

    # Update configuration
    update_strategy: UpdateStrategyName = UpdateStrategyName.ADJUSTMENT
    update_interval: int = 10
    reflection_enabled: bool = True
    continuous_learning: bool = True
    max_total_memories: int = 20000
    consolidation_threshold: float = 0.9
    namespace_migration_enabled: bool = True

    # System configuration
    enable_metrics: bool = True
    enable_distributed: bool = False
    redis_url: str | None = None
    kafka_brokers: list[str] | None = None

    # API keys (optional)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # Performance
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0
    batch_size: int = 20

    @classmethod
    def from_yaml(cls, path: Path) -> MemppSystemConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            config_dict: dict[str, Any] = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Write configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__.copy(), f, default_flow_style=False)


__all__ = ["MemppSystemConfig", "RetrievalStrategyName", "UpdateStrategyName"]
