import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, cast

import anthropic
import numpy as np
import torch
from openai import OpenAI
from pinecone import Pinecone, PodSpec, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============= Data Models =============


class TaskStatus(Enum):
    """Task completion status"""

    SUCCESS = auto()
    FAILURE = auto()
    PARTIAL = auto()
    TIMEOUT = auto()


class ActionType(Enum):
    """Types of actions an agent can take"""

    TOOL_USE = auto()
    REASONING = auto()
    OBSERVATION = auto()
    REFLECTION = auto()
    PLANNING = auto()


@dataclass
class Action:
    """Represents a single action in a trajectory"""

    type: ActionType
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the action."""
        return {
            "type": self.type.name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class State:
    """Represents environment state at a point in time"""

    description: str
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the state."""
        return {"description": self.description, "timestamp": self.timestamp.isoformat(), "attributes": self.attributes}


@dataclass
class Observation:
    """Environment observation after an action"""

    content: str
    timestamp: datetime
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the observation."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "reward": self.reward,
            "metadata": self.metadata,
        }


@dataclass
class Trajectory:
    """Complete interaction trajectory for a task"""

    task_id: str
    task_description: str
    states: list[State]
    actions: list[Action]
    observations: list[Observation]
    status: TaskStatus
    final_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of actions in the trajectory."""
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        """Sum of all observation rewards plus the final reward."""
        return sum(o.reward for o in self.observations if o.reward is not None) + self.final_reward

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the trajectory."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "states": [s.to_dict() for s in self.states],
            "actions": [a.to_dict() for a in self.actions],
            "observations": [o.to_dict() for o in self.observations],
            "status": self.status.name,
            "final_reward": self.final_reward,
            "metadata": self.metadata,
        }


class ProceduralMemory(BaseModel):
    """Base class for procedural memories"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_id: str
    task_pattern: str
    created_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    embedding: np.ndarray | None = None
    sparse_embedding: dict[str, float] | None = None  # For hybrid search
    metadata: dict[str, Any] = Field(default_factory=dict)

    def increment_usage(self, success: bool) -> None:
        """Update usage statistics"""
        self.usage_count += 1
        self.success_rate = (self.success_rate * (self.usage_count - 1) + (1.0 if success else 0.0)) / self.usage_count


class TrajectoryMemory(ProceduralMemory):
    """Memory storing raw trajectory.

    Notes:
        For built memories, `embedding` is always present. Override the
        field type from Optional to `np.ndarray` so static type checkers
        can rely on it being available at storage/query time.
    """

    trajectory: Trajectory
    key_states: list[State]
    critical_actions: list[Action]
    # Embedding is guaranteed for built memories
    embedding: np.ndarray | None = None


class ScriptMemory(ProceduralMemory):
    """Memory storing abstracted procedural script.

    As with `TrajectoryMemory`, ensure `embedding` is non-optional for
    downstream consumers and tighter type safety.
    """

    script: str
    steps: list[str]
    preconditions: list[str]
    postconditions: list[str]
    expected_outcomes: dict[str, Any]
    embedding: np.ndarray | None = None


class ProceduralizedMemory(ProceduralMemory):
    """Combined trajectory and script memory.

    The embedding here is the combination of the trajectory and script
    embeddings and is therefore guaranteed to be present.
    """

    trajectory: Trajectory
    script: str
    abstraction_level: float  # 0.0 = concrete, 1.0 = abstract
    key_patterns: list[dict[str, Any]]
    embedding: np.ndarray | None = None


# ============= Embedding Models =============


class EmbeddingModel(Protocol):
    """Protocol for embedding models"""

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Return embedding(s) for a string or list of strings."""
        ...


class MultilingualE5Embedder:
    """Multilingual E5 Large embedding model"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        if torch.backends.mps.is_built():
            device = "mps"

        self.model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.model.to(device)
        self.device = device

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode input text(s) into L2-normalized dense embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings


class OpenAIEmbedder:
    """OpenAI text-embedding-3-large model"""

    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-large"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode input text(s) using OpenAI embeddings API."""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(model=self.model, input=texts)

        embeddings = np.array([e.embedding for e in response.data])
        return embeddings


# ============= Memory Builders =============


class MemoryBuilder(ABC):
    """Abstract base class for memory builders"""

    def __init__(self, embedder: EmbeddingModel, sparse_encoder: BM25Encoder | None = None) -> None:
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder

    @abstractmethod
    async def build(self, trajectory: Trajectory) -> ProceduralMemory:
        """Build procedural memory from trajectory"""
        pass

    def _generate_memory_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_task_pattern(self, trajectory: Trajectory) -> str:
        """Extract task pattern from trajectory"""
        key_actions = [a.content for a in trajectory.actions[:3]]
        pattern = f"{trajectory.task_description} -> {' -> '.join(key_actions)}"
        return pattern

    def _create_sparse_embedding(self, text: str) -> dict[str, float]:
        """Create sparse embedding for hybrid search.

        Different versions of ``pinecone_text.BM25Encoder`` return slightly
        different shapes. Prefer a token->weight mapping, but gracefully
        handle a SparseVector-like object exposing ``indices`` and ``values``.
        """
        if not self.sparse_encoder:
            return {}

        raw: Any = self.sparse_encoder.encode_documents([text])

        # Common case: list[dict[str, float]]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return cast(dict[str, float], raw[0])

        # Fallback: SparseVector-like object with indices/values
        if hasattr(raw, "indices") and hasattr(raw, "values"):
            indices = getattr(raw, "indices")
            values = getattr(raw, "values")
            try:
                return {str(i): float(v) for i, v in zip(indices, values)}
            except Exception:
                return {}

        return {}


class TrajectoryBuilder(MemoryBuilder):
    """Builds memory by storing complete trajectories"""

    async def build(self, trajectory: Trajectory) -> TrajectoryMemory:
        """Build trajectory-based memory"""
        if trajectory.status != TaskStatus.SUCCESS:
            logger.warning(f"Building memory from non-successful trajectory: {trajectory.task_id}")

        # Extract key states and critical actions
        key_states = self._extract_key_states(trajectory)
        critical_actions = self._extract_critical_actions(trajectory)

        # Generate embedding from task description and key actions
        embed_text = f"{trajectory.task_description} {' '.join([a.content for a in critical_actions])}"
        embedding = self.embedder.encode(embed_text)[0]
        sparse_embedding = self._create_sparse_embedding(embed_text)

        memory = TrajectoryMemory(
            memory_id=self._generate_memory_id(trajectory.task_id),
            task_pattern=self._extract_task_pattern(trajectory),
            trajectory=trajectory,
            key_states=key_states,
            critical_actions=critical_actions,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            metadata={
                "source_task": trajectory.task_id,
                "trajectory_length": trajectory.length,
                "total_reward": trajectory.total_reward,
                "memory_type": "trajectory",
            },
        )

        return memory

    def _extract_key_states(self, trajectory: Trajectory) -> list[State]:
        """Extract important states from trajectory"""
        if len(trajectory.states) <= 3:
            return trajectory.states

        indices = [0, len(trajectory.states) // 2, -1]
        return [trajectory.states[i] for i in indices]

    def _extract_critical_actions(self, trajectory: Trajectory) -> list[Action]:
        """Extract critical actions from trajectory"""
        critical = [a for a in trajectory.actions if a.type in [ActionType.TOOL_USE, ActionType.PLANNING]]

        if len(critical) < 3:
            critical = trajectory.actions

        return critical[:10]


class ScriptBuilder(MemoryBuilder):
    """Builds memory by abstracting trajectories into scripts"""

    def __init__(
        self, embedder: EmbeddingModel, sparse_encoder: BM25Encoder | None = None, llm_client: Any | None = None
    ) -> None:
        super().__init__(embedder, sparse_encoder)
        self.llm_client = llm_client or anthropic.Anthropic()

    async def build(self, trajectory: Trajectory) -> ScriptMemory:
        """Build script-based memory through abstraction"""

        # Generate script from trajectory
        script = await self._generate_script(trajectory)
        steps = self._extract_steps(trajectory)
        preconditions = self._extract_preconditions(trajectory)
        postconditions = self._extract_postconditions(trajectory)

        # Generate embeddings
        embed_text = f"{trajectory.task_description} {script}"
        embedding = self.embedder.encode(embed_text)[0]
        sparse_embedding = self._create_sparse_embedding(embed_text)

        memory = ScriptMemory(
            memory_id=self._generate_memory_id(script),
            task_pattern=self._extract_task_pattern(trajectory),
            script=script,
            steps=steps,
            preconditions=preconditions,
            postconditions=postconditions,
            expected_outcomes={
                "success_indicators": self._extract_success_indicators(trajectory),
                "failure_patterns": self._extract_failure_patterns(trajectory),
            },
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            metadata={"source_task": trajectory.task_id, "abstraction_method": "llm_based", "memory_type": "script"},
        )

        return memory

    async def _generate_script(self, trajectory: Trajectory) -> str:
        """Generate abstract script using LLM"""
        prompt = f"""
        Analyze this task trajectory and create an abstract procedural script.

        Task: {trajectory.task_description}
        Actions taken: {[a.content for a in trajectory.actions[:10]]}
        Final status: {trajectory.status.name}

        Generate a concise, reusable script that captures the essential procedure.
        Focus on the general pattern, not specific details.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=500, messages=[{"role": "user", "content": prompt}]
        )

        # Only text blocks include ``text``; ignore tool/thinking blocks.
        blocks = getattr(response, "content", [])
        texts = [str(getattr(b, "text", "")) for b in blocks if getattr(b, "text", None)]
        return "\n".join(texts) if texts else ""

    def _extract_steps(self, trajectory: Trajectory) -> list[str]:
        """Extract high-level steps from trajectory"""
        steps = []
        for i, action in enumerate(trajectory.actions):
            if action.type in [ActionType.TOOL_USE, ActionType.PLANNING]:
                step = f"Step {i + 1}: {action.content}"
                steps.append(step)
        return steps

    def _extract_preconditions(self, trajectory: Trajectory) -> list[str]:
        """Extract task preconditions"""
        if trajectory.states:
            first_state = trajectory.states[0]
            return [f"Initial state: {first_state.description}"]
        return []

    def _extract_postconditions(self, trajectory: Trajectory) -> list[str]:
        """Extract task postconditions"""
        if trajectory.states:
            last_state = trajectory.states[-1]
            return [f"Final state: {last_state.description}"]
        return []

    def _extract_success_indicators(self, trajectory: Trajectory) -> list[str]:
        """Extract success indicators from successful trajectories"""
        if trajectory.status != TaskStatus.SUCCESS:
            return []

        indicators = []
        for obs in trajectory.observations:
            if obs.reward and obs.reward > 0:
                indicators.append(obs.content)
        return indicators[-3:] if indicators else []

    def _extract_failure_patterns(self, trajectory: Trajectory) -> list[str]:
        """Extract common failure patterns"""
        patterns = []
        for i, obs in enumerate(trajectory.observations):
            if (obs.reward and obs.reward < 0) and (i < len(trajectory.actions)):
                patterns.append(f"Avoid: {trajectory.actions[i].content}")
        return patterns[:3]


class ProceduralizationBuilder(MemoryBuilder):
    """Builds combined trajectory and script memory"""

    def __init__(
        self,
        embedder: EmbeddingModel,
        sparse_encoder: BM25Encoder | None = None,
        trajectory_builder: TrajectoryBuilder | None = None,
        script_builder: ScriptBuilder | None = None,
    ) -> None:
        super().__init__(embedder, sparse_encoder)
        self.trajectory_builder = trajectory_builder or TrajectoryBuilder(embedder, sparse_encoder)
        self.script_builder = script_builder or ScriptBuilder(embedder, sparse_encoder)

    async def build(self, trajectory: Trajectory) -> ProceduralizedMemory:
        """Build combined procedural memory"""

        # Build both types of memory
        trajectory_memory = await self.trajectory_builder.build(trajectory)
        script_memory = await self.script_builder.build(trajectory)

        # Extract key patterns by combining insights
        key_patterns = self._extract_key_patterns(trajectory_memory, script_memory)

        # Calculate abstraction level based on trajectory complexity
        abstraction_level = self._calculate_abstraction_level(trajectory)

        # Combine embeddings (weighted average)
        # Both embeddings are guaranteed (see memory classes), but guard defensively.
        t_emb = trajectory_memory.embedding
        s_emb = script_memory.embedding
        if t_emb is None or s_emb is None:
            raise ValueError("Missing embeddings to combine for proceduralized memory")
        combined_embedding = (t_emb + s_emb) / 2

        # Combine sparse embeddings
        combined_sparse: dict[str, float] = {}
        if trajectory_memory.sparse_embedding and script_memory.sparse_embedding:
            all_keys = set(trajectory_memory.sparse_embedding.keys()) | set(script_memory.sparse_embedding.keys())
            for key in all_keys:
                val1 = trajectory_memory.sparse_embedding.get(key, 0)
                val2 = script_memory.sparse_embedding.get(key, 0)
                combined_sparse[key] = (val1 + val2) / 2

        memory = ProceduralizedMemory(
            memory_id=self._generate_memory_id(f"{trajectory.task_id}_combined"),
            task_pattern=self._extract_task_pattern(trajectory),
            trajectory=trajectory,
            script=script_memory.script,
            abstraction_level=abstraction_level,
            key_patterns=key_patterns,
            embedding=combined_embedding,
            sparse_embedding=combined_sparse,
            metadata={
                "source_task": trajectory.task_id,
                "trajectory_length": trajectory.length,
                "script_steps": len(script_memory.steps),
                "combination_method": "weighted_average",
                "memory_type": "proceduralized",
            },
        )

        return memory

    def _extract_key_patterns(self, traj_mem: TrajectoryMemory, script_mem: ScriptMemory) -> list[dict[str, Any]]:
        """Extract key patterns from both memory types"""
        patterns = []

        for action in traj_mem.critical_actions[:3]:
            patterns.append({"type": "action_pattern", "content": action.content, "importance": "high"})

        for step in script_mem.steps[:3]:
            patterns.append({"type": "script_step", "content": step, "importance": "medium"})

        return patterns

    def _calculate_abstraction_level(self, trajectory: Trajectory) -> float:
        """Calculate appropriate abstraction level"""
        complexity_score = len(trajectory.actions) / 50.0
        return min(1.0, complexity_score)


# ============= Pinecone Memory Storage =============


class PineconeMemoryStorage:
    """Manages storage and retrieval using Pinecone"""

    def __init__(
        self,
        api_key: str,
        environment: str = "us-east-1",
        index_name: str = "mempp-memories",
        dimension: int = 1024,
        metric: str = "dotproduct",
        use_serverless: bool = True,
    ) -> None:
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension

        # Create or connect to index
        self._init_index(environment, metric, use_serverless)

        # Initialize sparse encoder for hybrid search
        self.sparse_encoder = BM25Encoder()
        self.sparse_encoder.fit(["sample text for initialization"])

        # Local cache for memory objects
        self.memories: dict[str, ProceduralMemory] = {}

        logger.info(f"Pinecone storage initialized with index: {index_name}")

    def _init_index(self, environment: str, metric: str, use_serverless: bool) -> None:
        """Initialize Pinecone index"""

        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            # Create new index
            spec: ServerlessSpec | PodSpec
            if use_serverless:
                spec = ServerlessSpec(cloud="aws", region=environment)
            else:
                spec = PodSpec(environment=environment, pod_type="p1.x1", pods=1)

            self.pc.create_index(name=self.index_name, dimension=self.dimension, metric=metric, spec=spec)

            logger.info(f"Created new Pinecone index: {self.index_name}")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

    async def store(self, memory: ProceduralMemory, namespace: str | None = None) -> str:
        """Store procedural memory in Pinecone"""

        memory_id = memory.memory_id

        # Determine namespace based on memory type
        if namespace is None:
            if isinstance(memory, TrajectoryMemory):
                namespace = "trajectory"
            elif isinstance(memory, ScriptMemory):
                namespace = "script"
            elif isinstance(memory, ProceduralizedMemory):
                namespace = "proceduralized"
            else:
                namespace = "default"

        # Prepare metadata for Pinecone
        metadata = {
            "memory_type": memory.metadata.get("memory_type", "unknown"),
            "task_pattern": memory.task_pattern[:500],  # Pinecone metadata size limit
            "created_at": memory.created_at.isoformat(),
            "usage_count": memory.usage_count,
            "success_rate": memory.success_rate,
            "source_task": memory.metadata.get("source_task", ""),
            "total_reward": memory.metadata.get("total_reward", 0.0),
        }

        # Prepare vector for upsert
        vectors = []

        # Ensure embedding exists before serializing
        if memory.embedding is None:
            raise ValueError("Embedding is required to store memory in Pinecone")

        # Handle hybrid search (dense + sparse vectors)
        if memory.sparse_embedding:
            # Hybrid vector format for Pinecone
            vectors.append({
                "id": memory_id,
                "values": memory.embedding.tolist(),
                "sparse_values": {
                    "indices": list(memory.sparse_embedding.keys()),
                    "values": list(memory.sparse_embedding.values()),
                },
                "metadata": metadata,
            })
        else:
            # Dense vector only
            vectors.append({"id": memory_id, "values": memory.embedding.tolist(), "metadata": metadata})

        # Upsert to Pinecone
        self.index.upsert(vectors=vectors, namespace=namespace)

        # Store memory object locally
        self.memories[memory_id] = memory

        # Persist to disk as backup
        await self._persist_memory(memory)

        logger.info(
            f"Stored {memory.metadata.get('memory_type', 'unknown')} memory: {memory_id} in namespace: {namespace}"
        )
        return memory_id

    async def _persist_memory(self, memory: ProceduralMemory) -> None:
        """Persist memory to disk as backup"""
        storage_path = Path("./mempp_storage")
        storage_path.mkdir(parents=True, exist_ok=True)

        memory_file = storage_path / f"{memory.memory_id}.json"

        # Convert to serializable format
        memory_dict = memory.model_dump(exclude={"embedding", "sparse_embedding"})

        # Save embeddings separately
        embedding_file = storage_path / f"{memory.memory_id}.npy"
        if memory.embedding is None:
            raise ValueError("Embedding is required to persist memory to disk")
        np.save(embedding_file, memory.embedding)

        with open(memory_file, "w") as f:
            json.dump(memory_dict, f, indent=2, default=str)

    def query(
        self,
        query_vector: np.ndarray,
        namespace: str = "proceduralized",
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        sparse_vector: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Query Pinecone index"""

        # Prepare query
        query_params = {
            "namespace": namespace,
            "top_k": top_k,
            "include_values": False,
            "include_metadata": include_metadata,
            "vector": query_vector.tolist(),
        }

        # Add sparse vector for hybrid search
        if sparse_vector:
            query_params["sparse_vector"] = {
                "indices": list(sparse_vector.keys()),
                "values": list(sparse_vector.values()),
            }

        # Add metadata filter
        if filter:
            query_params["filter"] = filter

        # Execute query
        results: Any = self.index.query(**query_params)

        matches = getattr(results, "matches", results)
        return cast(list[dict[str, Any]], matches)

    def delete(self, memory_id: str, namespace: str = "proceduralized") -> None:
        """Delete memory from Pinecone"""
        self.index.delete(ids=[memory_id], namespace=namespace)

        if memory_id in self.memories:
            del self.memories[memory_id]

        logger.info(f"Deleted memory: {memory_id} from namespace: {namespace}")

    def update_metadata(self, memory_id: str, metadata: dict[str, Any], namespace: str = "proceduralized") -> None:
        """Update memory metadata in Pinecone"""
        self.index.update(id=memory_id, set_metadata=metadata, namespace=namespace)

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics from Pinecone"""
        stats = self.index.describe_index_stats()

        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": stats.namespaces,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "local_memories": len(self.memories),
        }


# ============= Main Build Pipeline =============


class MemppBuildPipeline:
    """Main pipeline for building procedural memories with Pinecone"""

    def __init__(
        self,
        pinecone_api_key: str,
        embedder: EmbeddingModel | None = None,
        storage: PineconeMemoryStorage | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.embedder: EmbeddingModel = embedder or MultilingualE5Embedder()
        self.storage = storage or PineconeMemoryStorage(api_key=pinecone_api_key)
        self.llm_client = llm_client

        # Initialize sparse encoder
        self.sparse_encoder = BM25Encoder()

        # Initialize builders
        self.trajectory_builder = TrajectoryBuilder(self.embedder, self.sparse_encoder)
        self.script_builder = ScriptBuilder(self.embedder, self.sparse_encoder, self.llm_client)
        self.proceduralization_builder = ProceduralizationBuilder(
            self.embedder, self.sparse_encoder, self.trajectory_builder, self.script_builder
        )

        self.build_stats: dict[str, Any] = {"total_processed": 0, "successful_builds": 0, "failed_builds": 0, "build_times": []}

    async def build_from_trajectory(
        self, trajectory: Trajectory, strategy: str = "proceduralization"
    ) -> ProceduralMemory:
        """Build procedural memory from a single trajectory"""

        start_time = datetime.now()

        try:
            memory: ProceduralMemory
            match strategy:
                case "trajectory":
                    memory = await self.trajectory_builder.build(trajectory)
                    namespace = "trajectory"

                case "script":
                    memory = await self.script_builder.build(trajectory)
                    namespace = "script"

                case "proceduralization":
                    memory = await self.proceduralization_builder.build(trajectory)
                    namespace = "proceduralized"

                case _:
                    raise ValueError(f"Unknown build strategy: {strategy}")

            # Store in Pinecone
            await self.storage.store(memory, namespace=namespace)

            # Update statistics
            build_time = (datetime.now() - start_time).total_seconds()
            self.build_stats["total_processed"] += 1
            self.build_stats["successful_builds"] += 1
            self.build_stats["build_times"].append(build_time)

            logger.info(f"Successfully built {strategy} memory in {build_time:.2f}s")
            return memory

        except Exception as e:
            self.build_stats["total_processed"] += 1
            self.build_stats["failed_builds"] += 1
            logger.error(f"Failed to build memory: {e}")
            raise

    async def batch_build(
        self, trajectories: list[Trajectory], strategy: str = "proceduralization", filter_successful: bool = True
    ) -> list[ProceduralMemory]:
        """Build memories from multiple trajectories"""

        if filter_successful:
            trajectories = [t for t in trajectories if t.status == TaskStatus.SUCCESS]
            logger.info(f"Filtered to {len(trajectories)} successful trajectories")

        # Train sparse encoder on batch
        if self.sparse_encoder:
            texts = [t.task_description for t in trajectories]
            self.sparse_encoder.fit(texts)

        memories = []
        for trajectory in trajectories:
            try:
                memory = await self.build_from_trajectory(trajectory, strategy)
                memories.append(memory)
            except Exception as e:
                logger.warning(f"Skipping trajectory {trajectory.task_id}: {e}")
                continue

        logger.info(f"Built {len(memories)} memories from {len(trajectories)} trajectories")
        return memories

    def get_build_statistics(self) -> dict[str, Any]:
        """Get build pipeline statistics"""
        avg_build_time = (
            sum(self.build_stats["build_times"]) / len(self.build_stats["build_times"])
            if self.build_stats["build_times"]
            else 0
        )

        return {
            **self.build_stats,
            "average_build_time": avg_build_time,
            "success_rate": (
                self.build_stats["successful_builds"] / self.build_stats["total_processed"]
                if self.build_stats["total_processed"] > 0
                else 0
            ),
            "storage_stats": self.storage.get_statistics(),
        }
