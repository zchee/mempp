import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import anthropic
import numpy as np
from prometheus_client import Counter, Gauge, Histogram
from sklearn.cluster import DBSCAN

# Import from previous components
from mempp.build import (
    EmbeddingModel,
    MemppBuildPipeline,
    PineconeMemoryStorage,
    ProceduralizedMemory,
    ProceduralMemory,
    ScriptMemory,
    TaskStatus,
    Trajectory,
    TrajectoryMemory,
)
from mempp.retrieve import (
    MemppRetrievalPipeline,
    RetrievalResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============= Update Strategies =============


class UpdateStrategy(Enum):
    """Different memory update strategies"""

    VANILLA = auto()  # Simple addition
    VALIDATION = auto()  # Only successful trajectories
    ADJUSTMENT = auto()  # Reflection-based correction
    DEPRECATION = auto()  # Time-based decay
    CONSOLIDATION = auto()  # Memory merging
    PRUNING = auto()  # Remove low-value memories
    REINFORCEMENT = auto()  # Strengthen successful patterns
    NAMESPACE_MIGRATION = auto()  # Move memories between namespaces


@dataclass
class UpdateConfig:
    """Configuration for memory updates"""

    batch_size: int = 10
    success_threshold: float = 0.7
    deprecation_rate: float = 0.95  # Per update cycle
    min_usage_for_keep: int = 2
    max_memory_per_namespace: int = 5000  # Per Pinecone namespace
    max_total_memories: int = 20000
    consolidation_threshold: float = 0.9  # Similarity for merging
    reflection_enabled: bool = True
    continuous_learning: bool = True
    update_interval: int = 100  # Tasks between updates
    enable_metrics: bool = True
    namespace_rebalancing: bool = True  # Enable Pinecone namespace optimization
    auto_index_optimization: bool = True  # Enable Pinecone index optimization


@dataclass
class UpdateResult:
    """Result of a memory update operation"""

    operation: str
    memory_id: str
    success: bool
    timestamp: datetime
    namespace: str = "proceduralized"
    changes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryHealth:
    """Health metrics for a memory"""

    memory_id: str
    namespace: str
    usage_count: int
    success_rate: float
    last_used: datetime
    age_days: float
    health_score: float
    recommendation: str  # keep, adjust, deprecate, remove, migrate


# ============= Reflection Engine =============


class ReflectionEngine:
    """Engine for reflecting on and correcting failed experiences"""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client or anthropic.Anthropic()
        self.reflection_history = deque(maxlen=100)

    async def reflect_on_failure(
        self,
        original_memory: ProceduralMemory,
        failed_trajectory: Trajectory,
        retrieved_memory: RetrievalResult,
    ) -> dict[str, Any]:
        """Reflect on why a memory led to failure"""

        reflection_prompt = f"""
        Analyze this failure case where a procedural memory led to an incorrect execution.
        
        Original Memory Pattern: {original_memory.task_pattern}
        Memory Type: {type(original_memory).__name__}
        Retrieved from Namespace: {retrieved_memory.namespace}
        
        Task Attempted: {failed_trajectory.task_description}
        Final Status: {failed_trajectory.status.name}
        Actions Taken: {[a.content for a in failed_trajectory.actions[:10]]}
        
        Identify:
        1. Why the memory was incorrectly retrieved (similarity: {retrieved_memory.score:.3f})
        2. What went wrong in the execution
        3. How to correct the memory
        4. Whether this memory should be adjusted, deprecated, migrated to another namespace, or kept with warnings
        5. If the memory should be in a different namespace for better retrieval
        
        Provide structured analysis and correction suggestions.
        """

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": reflection_prompt}],
            )

            # Collect text blocks only; ignore tool/thinking blocks
            blocks = getattr(response, "content", [])
            text = "\n".join([str(getattr(b, "text", "")) for b in blocks if getattr(b, "text", None)])
            reflection = self._parse_reflection(text)

            # Store in history
            self.reflection_history.append({
                "timestamp": datetime.now(),
                "memory_id": original_memory.memory_id,
                "namespace": retrieved_memory.namespace,
                "task_id": failed_trajectory.task_id,
                "reflection": reflection,
            })

            return reflection

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"error": str(e), "recommendation": "keep", "corrections": []}

    async def generate_correction(self, memory: ProceduralMemory, reflection: dict[str, Any]) -> ProceduralMemory:
        """Generate corrected version of memory based on reflection"""

        if isinstance(memory, ScriptMemory):
            corrected = await self._correct_script_memory(memory, reflection)
        elif isinstance(memory, TrajectoryMemory):
            corrected = await self._correct_trajectory_memory(memory, reflection)
        elif isinstance(memory, ProceduralizedMemory):
            corrected = await self._correct_proceduralized_memory(memory, reflection)
        else:
            corrected = memory

        # Update metadata
        corrected.metadata["corrected"] = True
        corrected.metadata["correction_timestamp"] = datetime.now().isoformat()
        corrected.metadata["reflection"] = reflection

        return corrected

    async def _correct_script_memory(self, memory: ScriptMemory, reflection: dict[str, Any]) -> ScriptMemory:
        """Correct a script-based memory"""

        correction_prompt = f"""
        Correct this procedural script based on the reflection analysis.
        
        Original Script: {memory.script}
        Steps: {memory.steps}
        
        Issues Identified: {reflection.get("issues", [])}
        Corrections Needed: {reflection.get("corrections", [])}
        
        Generate a corrected script that addresses these issues.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": correction_prompt}],
        )

        blocks = getattr(response, "content", [])
        corrected_script = "\n".join([str(getattr(b, "text", "")) for b in blocks if getattr(b, "text", None)])

        # Create corrected memory
        corrected = ScriptMemory(
            memory_id=f"{memory.memory_id}_corrected",
            task_pattern=memory.task_pattern,
            script=corrected_script,
            steps=self._extract_steps_from_script(corrected_script),
            preconditions=memory.preconditions,
            postconditions=memory.postconditions,
            expected_outcomes=memory.expected_outcomes,
            embedding=memory.embedding,  # Will be updated later
            sparse_embedding=memory.sparse_embedding,
            metadata=memory.metadata.copy(),
        )

        return corrected

    async def _correct_trajectory_memory(
        self, memory: TrajectoryMemory, reflection: dict[str, Any]
    ) -> TrajectoryMemory:
        """Correct a trajectory-based memory"""

        corrected = TrajectoryMemory(
            memory_id=f"{memory.memory_id}_corrected",
            task_pattern=memory.task_pattern,
            trajectory=memory.trajectory,
            key_states=memory.key_states,
            critical_actions=memory.critical_actions,
            embedding=memory.embedding,
            sparse_embedding=memory.sparse_embedding,
            metadata=memory.metadata.copy(),
        )

        # Add warning annotations
        corrected.metadata["warnings"] = reflection.get("warnings", [])
        corrected.metadata["avoid_patterns"] = reflection.get("avoid_patterns", [])

        return corrected

    async def _correct_proceduralized_memory(
        self, memory: ProceduralizedMemory, reflection: dict[str, Any]
    ) -> ProceduralizedMemory:
        """Correct a combined memory"""

        # Correct the script portion
        script_correction = await self._correct_script_memory(
            ScriptMemory(
                memory_id=memory.memory_id,
                task_pattern=memory.task_pattern,
                script=memory.script,
                steps=[],
                preconditions=[],
                postconditions=[],
                expected_outcomes={},
                embedding=memory.embedding,
                sparse_embedding=memory.sparse_embedding,
            ),
            reflection,
        )

        corrected = ProceduralizedMemory(
            memory_id=f"{memory.memory_id}_corrected",
            task_pattern=memory.task_pattern,
            trajectory=memory.trajectory,
            script=script_correction.script,
            abstraction_level=memory.abstraction_level,
            key_patterns=memory.key_patterns,
            embedding=memory.embedding,
            sparse_embedding=memory.sparse_embedding,
            metadata=memory.metadata.copy(),
        )

        return corrected

    def _parse_reflection(self, text: str) -> dict[str, Any]:
        """Parse reflection text into structured format"""

        reflection = {
            "issues": [],
            "corrections": [],
            "warnings": [],
            "avoid_patterns": [],
            "recommendation": "adjust",
            "namespace_suggestion": None,
        }

        lines = text.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if "issue" in line.lower() or "problem" in line.lower():
                current_section = "issues"
            elif "correct" in line.lower() or "fix" in line.lower():
                current_section = "corrections"
            elif "warn" in line.lower() or "caution" in line.lower():
                current_section = "warnings"
            elif "avoid" in line.lower():
                current_section = "avoid_patterns"
            elif "namespace" in line.lower():
                # Extract namespace suggestion
                if "script" in line.lower():
                    reflection["namespace_suggestion"] = "script"
                elif "trajectory" in line.lower():
                    reflection["namespace_suggestion"] = "trajectory"
                elif "procedural" in line.lower():
                    reflection["namespace_suggestion"] = "proceduralized"
            elif "recommend" in line.lower():
                if "deprecate" in line.lower():
                    reflection["recommendation"] = "deprecate"
                elif "remove" in line.lower():
                    reflection["recommendation"] = "remove"
                elif "migrate" in line.lower():
                    reflection["recommendation"] = "migrate"
                elif "adjust" in line.lower():
                    reflection["recommendation"] = "adjust"
            elif current_section and line:
                reflection[current_section].append(line)

        return reflection

    def _extract_steps_from_script(self, script: str) -> list[str]:
        """Extract steps from corrected script"""
        steps = []
        for line in script.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.", "-", "*", "Step")):
                steps.append(line.strip())
        return steps


# ============= Pinecone Memory Consolidator =============


class PineconeMemoryConsolidator:
    """Consolidates similar memories in Pinecone to reduce redundancy"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        embedder: EmbeddingModel,
        similarity_threshold: float = 0.9,
    ):
        self.storage = storage
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    async def find_similar_memories_in_namespace(
        self, namespace: str, sample_size: int = 100
    ) -> list[list[ProceduralMemory]]:
        """Find groups of similar memories within a namespace"""

        # Sample memories from namespace
        sample_vector = np.random.randn(1024)  # Random vector for sampling
        matches = self.storage.query(query_vector=sample_vector, namespace=namespace, top_k=sample_size)

        if len(matches) < 2:
            return []

        # Get memory objects
        memories = []
        embeddings = []
        for match in matches:
            if match["id"] in self.storage.memories:
                memory = self.storage.memories[match["id"]]
                memories.append(memory)
                embeddings.append(memory.embedding)

        if len(memories) < 2:
            return []

        embeddings = np.array(embeddings)

        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=1 - self.similarity_threshold, min_samples=2, metric="cosine").fit(embeddings)

        # Group memories by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise points
                clusters[label].append(memories[i])

        return list(clusters.values())

    async def consolidate_group(self, similar_memories: list[ProceduralMemory], namespace: str) -> ProceduralMemory:
        """Consolidate a group of similar memories into one"""

        if len(similar_memories) == 1:
            return similar_memories[0]

        # Sort by success rate and usage
        similar_memories.sort(key=lambda m: (m.success_rate, m.usage_count), reverse=True)

        base_memory = similar_memories[0]

        # Merge statistics
        total_usage = sum(m.usage_count for m in similar_memories)
        weighted_success = (
            sum(m.success_rate * m.usage_count for m in similar_memories) / total_usage if total_usage > 0 else 0
        )

        # Create consolidated memory (narrow the list type before passing)
        if isinstance(base_memory, ScriptMemory):
            script_mems = [m for m in similar_memories if isinstance(m, ScriptMemory)]
            consolidated = await self._consolidate_scripts(script_mems)
        elif isinstance(base_memory, TrajectoryMemory):
            traj_mems = [m for m in similar_memories if isinstance(m, TrajectoryMemory)]
            consolidated = await self._consolidate_trajectories(traj_mems)
        elif isinstance(base_memory, ProceduralizedMemory):
            proc_mems = [m for m in similar_memories if isinstance(m, ProceduralizedMemory)]
            consolidated = await self._consolidate_proceduralized(proc_mems)
        else:
            consolidated = base_memory

        # Update statistics
        consolidated.usage_count = total_usage
        consolidated.success_rate = weighted_success
        consolidated.metadata["consolidated"] = True
        consolidated.metadata["source_memories"] = [m.memory_id for m in similar_memories]
        consolidated.metadata["consolidation_timestamp"] = datetime.now().isoformat()
        consolidated.metadata["namespace"] = namespace

        return consolidated

    async def _consolidate_scripts(self, memories: list[ScriptMemory]) -> ScriptMemory:
        """Consolidate script memories"""

        # Combine all steps and deduplicate
        all_steps = []
        for memory in memories:
            all_steps.extend(memory.steps)

        seen = set()
        unique_steps = []
        for step in all_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)

        # Combine scripts
        combined_script = "\n".join([m.script for m in memories[:3]])  # Top 3

        # Average embeddings
        avg_embedding = np.mean([m.embedding for m in memories], axis=0)

        # Combine sparse embeddings
        combined_sparse = {}
        for memory in memories:
            if memory.sparse_embedding:
                for key, value in memory.sparse_embedding.items():
                    if key in combined_sparse:
                        combined_sparse[key] = max(combined_sparse[key], value)
                    else:
                        combined_sparse[key] = value

        consolidated = ScriptMemory(
            memory_id=f"consolidated_{memories[0].memory_id[:8]}",
            task_pattern=memories[0].task_pattern,
            script=combined_script,
            steps=unique_steps[:10],
            preconditions=list(set(sum([m.preconditions for m in memories], []))),
            postconditions=list(set(sum([m.postconditions for m in memories], []))),
            expected_outcomes=memories[0].expected_outcomes,
            embedding=avg_embedding,
            sparse_embedding=combined_sparse,
        )

        return consolidated

    async def _consolidate_trajectories(self, memories: list[TrajectoryMemory]) -> TrajectoryMemory:
        """Consolidate trajectory memories"""

        best_memory = max(memories, key=lambda m: (m.success_rate, m.usage_count))

        # Combine critical actions from all
        all_critical_actions = []
        for memory in memories:
            all_critical_actions.extend(memory.critical_actions)

        seen_actions = set()
        unique_actions = []
        for action in all_critical_actions:
            action_str = f"{action.type}_{action.content}"
            if action_str not in seen_actions:
                seen_actions.add(action_str)
                unique_actions.append(action)

        # Average embeddings
        avg_embedding = np.mean([m.embedding for m in memories], axis=0)

        # Combine sparse embeddings
        combined_sparse = {}
        for memory in memories:
            if memory.sparse_embedding:
                for key, value in memory.sparse_embedding.items():
                    if key in combined_sparse:
                        combined_sparse[key] = max(combined_sparse[key], value)
                    else:
                        combined_sparse[key] = value

        consolidated = TrajectoryMemory(
            memory_id=f"consolidated_{best_memory.memory_id[:8]}",
            task_pattern=best_memory.task_pattern,
            trajectory=best_memory.trajectory,
            key_states=best_memory.key_states,
            critical_actions=unique_actions[:15],
            embedding=avg_embedding,
            sparse_embedding=combined_sparse,
        )

        return consolidated

    async def _consolidate_proceduralized(self, memories: list[ProceduralizedMemory]) -> ProceduralizedMemory:
        """Consolidate proceduralized memories"""

        # Consolidate scripts
        script_memories = [
            ScriptMemory(
                memory_id=m.memory_id,
                task_pattern=m.task_pattern,
                script=m.script,
                steps=[],
                preconditions=[],
                postconditions=[],
                expected_outcomes={},
                embedding=m.embedding,
                sparse_embedding=m.sparse_embedding,
            )
            for m in memories
        ]

        consolidated_script = await self._consolidate_scripts(script_memories)

        best_memory = max(memories, key=lambda m: (m.success_rate, m.usage_count))

        # Average embeddings
        avg_embedding = np.mean([m.embedding for m in memories], axis=0)

        # Combine sparse embeddings
        combined_sparse = {}
        for memory in memories:
            if memory.sparse_embedding:
                for key, value in memory.sparse_embedding.items():
                    if key in combined_sparse:
                        combined_sparse[key] = max(combined_sparse[key], value)
                    else:
                        combined_sparse[key] = value

        consolidated = ProceduralizedMemory(
            memory_id=f"consolidated_{best_memory.memory_id[:8]}",
            task_pattern=best_memory.task_pattern,
            trajectory=best_memory.trajectory,
            script=consolidated_script.script,
            abstraction_level=float(np.mean([m.abstraction_level for m in memories])),
            key_patterns=best_memory.key_patterns,
            embedding=avg_embedding,
            sparse_embedding=combined_sparse,
        )

        return consolidated


# ============= Pinecone Memory Health Monitor =============


class PineconeMemoryHealthMonitor:
    """Monitors and evaluates memory health in Pinecone"""

    def __init__(self, storage: PineconeMemoryStorage, config: UpdateConfig):
        self.storage = storage
        self.config = config
        self.health_history = defaultdict(list)

        if config.enable_metrics:
            # Prometheus metrics
            self.memory_health_gauge = Gauge(
                "memory_health_score",
                "Health score of procedural memories",
                ["memory_id", "namespace"],
            )
            self.namespace_usage_gauge = Gauge(
                "namespace_memory_count",
                "Number of memories per namespace",
                ["namespace"],
            )

    def evaluate_memory_health(self, memory: ProceduralMemory, namespace: str) -> MemoryHealth:
        """Evaluate the health of a memory"""

        # Calculate age
        age_days = (datetime.now() - memory.created_at).days

        # Calculate last used (from metadata if available)
        last_used = memory.metadata.get("last_used", memory.created_at)
        if isinstance(last_used, str):
            last_used = datetime.fromisoformat(last_used)

        # Calculate health score (0-1)
        health_score = self._calculate_health_score(
            success_rate=memory.success_rate,
            usage_count=memory.usage_count,
            age_days=age_days,
            days_since_used=(datetime.now() - last_used).days,
        )

        # Determine recommendation
        recommendation = self._get_recommendation(
            health_score=health_score,
            usage_count=memory.usage_count,
            success_rate=memory.success_rate,
            namespace=namespace,
        )

        health = MemoryHealth(
            memory_id=memory.memory_id,
            namespace=namespace,
            usage_count=memory.usage_count,
            success_rate=memory.success_rate,
            last_used=last_used,
            age_days=age_days,
            health_score=health_score,
            recommendation=recommendation,
        )

        # Update metrics
        if self.config.enable_metrics:
            self.memory_health_gauge.labels(memory_id=memory.memory_id, namespace=namespace).set(health_score)

        # Store history
        self.health_history[memory.memory_id].append({
            "timestamp": datetime.now(),
            "namespace": namespace,
            "health_score": health_score,
            "recommendation": recommendation,
        })

        return health

    def _calculate_health_score(
        self, success_rate: float, usage_count: int, age_days: int, days_since_used: int
    ) -> float:
        """Calculate health score based on multiple factors"""

        # Success rate component (40%)
        success_score = success_rate * 0.4

        # Usage frequency component (30%)
        usage_score = min(1.0, usage_count / 10) * 0.3

        # Recency component (20%)
        recency_score = max(0, 1.0 - (days_since_used / 30)) * 0.2

        # Age penalty component (10%)
        age_score = max(0, 1.0 - (age_days / 365)) * 0.1

        return success_score + usage_score + recency_score + age_score

    def _get_recommendation(self, health_score: float, usage_count: int, success_rate: float, namespace: str) -> str:
        """Get recommendation based on health metrics"""

        if health_score < 0.3:
            return "remove"
        elif health_score < 0.5:
            if success_rate < 0.5:
                return "adjust"
            else:
                return "deprecate"
        elif usage_count < self.config.min_usage_for_keep:
            return "deprecate"
        elif namespace == "trajectory" and health_score > 0.7:
            # Consider migrating successful trajectories to proceduralized
            return "migrate"
        else:
            return "keep"

    async def analyze_namespace_health(self, namespace: str) -> dict[str, Any]:
        """Analyze overall health of a namespace"""

        # Get namespace statistics from Pinecone
        stats = self.storage.index.describe_index_stats()
        namespace_stats = stats.namespaces.get(namespace, {})

        if self.config.enable_metrics:
            self.namespace_usage_gauge.labels(namespace=namespace).set(namespace_stats.get("vector_count", 0))

        return {
            "namespace": namespace,
            "vector_count": namespace_stats.get("vector_count", 0),
            "recommendation": self._get_namespace_recommendation(namespace_stats),
        }

    def _get_namespace_recommendation(self, namespace_stats: dict) -> str:
        """Get recommendation for namespace management"""

        vector_count = namespace_stats.get("vector_count", 0)

        if vector_count > self.config.max_memory_per_namespace:
            return "needs_pruning"
        elif vector_count < 10:
            return "underutilized"
        else:
            return "healthy"


# ============= Update Executors =============


class PineconeUpdateExecutor(ABC):
    """Abstract base for Pinecone update execution"""

    @abstractmethod
    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Execute update operation"""
        pass


class PineconeVanillaUpdateExecutor(PineconeUpdateExecutor):
    """Simple addition of new memories to Pinecone"""

    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Add all memories without filtering"""

        results = []

        for memory in memories_to_update:
            try:
                # Determine namespace based on memory type
                if isinstance(memory, TrajectoryMemory):
                    namespace = "trajectory"
                elif isinstance(memory, ScriptMemory):
                    namespace = "script"
                else:
                    namespace = "proceduralized"

                memory_id = await storage.store(memory, namespace=namespace)

                results.append(
                    UpdateResult(
                        operation="add",
                        memory_id=memory_id,
                        success=True,
                        timestamp=datetime.now(),
                        namespace=namespace,
                        changes={"added": True},
                    )
                )
            except Exception as e:
                logger.error(f"Failed to add memory to Pinecone: {e}")
                results.append(
                    UpdateResult(
                        operation="add",
                        memory_id=memory.memory_id,
                        success=False,
                        timestamp=datetime.now(),
                        namespace="unknown",
                        changes={"error": str(e)},
                    )
                )

        return results


class PineconeValidationUpdateExecutor(PineconeUpdateExecutor):
    """Add only successful memories to Pinecone"""

    def __init__(self, success_threshold: float = 0.0):
        self.success_threshold = success_threshold

    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Add only memories from successful trajectories"""

        results = []
        trajectories = context.get("trajectories", [])

        for memory, trajectory in zip(memories_to_update, trajectories, strict=False):
            if trajectory.status == TaskStatus.SUCCESS and trajectory.final_reward >= self.success_threshold:
                try:
                    # Determine namespace
                    if isinstance(memory, TrajectoryMemory):
                        namespace = "trajectory"
                    elif isinstance(memory, ScriptMemory):
                        namespace = "script"
                    else:
                        namespace = "proceduralized"

                    memory_id = await storage.store(memory, namespace=namespace)

                    results.append(
                        UpdateResult(
                            operation="add_validated",
                            memory_id=memory_id,
                            success=True,
                            timestamp=datetime.now(),
                            namespace=namespace,
                            changes={
                                "validated": True,
                                "reward": trajectory.final_reward,
                            },
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to add validated memory: {e}")
                    results.append(
                        UpdateResult(
                            operation="add_validated",
                            memory_id=memory.memory_id,
                            success=False,
                            timestamp=datetime.now(),
                            namespace="unknown",
                            changes={"error": str(e)},
                        )
                    )
            else:
                results.append(
                    UpdateResult(
                        operation="skip_validation",
                        memory_id=memory.memory_id,
                        success=False,
                        timestamp=datetime.now(),
                        namespace="unknown",
                        changes={
                            "reason": "failed_validation",
                            "status": trajectory.status.name,
                        },
                    )
                )

        return results


class PineconeNamespaceMigrationExecutor(PineconeUpdateExecutor):
    """Migrate memories between Pinecone namespaces"""

    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Migrate memories between namespaces based on performance"""

        results = []
        migrations = context.get("migrations", [])

        for migration in migrations:
            memory = migration["memory"]
            from_namespace = migration["from_namespace"]
            to_namespace = migration["to_namespace"]

            try:
                # Delete from old namespace
                storage.delete(memory.memory_id, namespace=from_namespace)

                # Add to new namespace
                await storage.store(memory, namespace=to_namespace)

                results.append(
                    UpdateResult(
                        operation="migrate",
                        memory_id=memory.memory_id,
                        success=True,
                        timestamp=datetime.now(),
                        namespace=to_namespace,
                        changes={
                            "from_namespace": from_namespace,
                            "to_namespace": to_namespace,
                        },
                    )
                )

                logger.info(f"Migrated {memory.memory_id} from {from_namespace} to {to_namespace}")

            except Exception as e:
                logger.error(f"Failed to migrate memory: {e}")
                results.append(
                    UpdateResult(
                        operation="migrate",
                        memory_id=memory.memory_id,
                        success=False,
                        timestamp=datetime.now(),
                        namespace=from_namespace,
                        changes={"error": str(e)},
                    )
                )

        return results


class PineconeConsolidationUpdateExecutor(PineconeUpdateExecutor):
    """Consolidate similar memories in Pinecone"""

    def __init__(self, consolidator: PineconeMemoryConsolidator):
        self.consolidator = consolidator

    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Find and consolidate similar memories within namespaces"""

        results = []

        # Process each namespace
        for namespace in ["proceduralized", "script", "trajectory"]:
            # Find similar memory groups in this namespace
            similar_groups = await self.consolidator.find_similar_memories_in_namespace(namespace, sample_size=100)

            for group in similar_groups:
                if len(group) < 2:
                    continue

                # Consolidate group
                consolidated = await self.consolidator.consolidate_group(group, namespace)

                try:
                    # Store consolidated memory
                    memory_id = await storage.store(consolidated, namespace=namespace)

                    # Remove original memories from Pinecone
                    for memory in group:
                        storage.delete(memory.memory_id, namespace=namespace)

                    results.append(
                        UpdateResult(
                            operation="consolidate",
                            memory_id=memory_id,
                            success=True,
                            timestamp=datetime.now(),
                            namespace=namespace,
                            changes={
                                "consolidated_count": len(group),
                                "source_memories": [m.memory_id for m in group],
                            },
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to consolidate memories: {e}")
                    results.append(
                        UpdateResult(
                            operation="consolidate",
                            memory_id=consolidated.memory_id,
                            success=False,
                            timestamp=datetime.now(),
                            namespace=namespace,
                            changes={"error": str(e)},
                        )
                    )

        return results


class PineconePruningUpdateExecutor(PineconeUpdateExecutor):
    """Prune low-value memories from Pinecone"""

    def __init__(self, health_monitor: PineconeMemoryHealthMonitor):
        self.health_monitor = health_monitor

    async def execute(
        self,
        storage: PineconeMemoryStorage,
        memories_to_update: list[ProceduralMemory],
        context: dict[str, Any],
    ) -> list[UpdateResult]:
        """Remove unhealthy memories from each namespace"""

        results = []

        for namespace in ["proceduralized", "script", "trajectory"]:
            # Analyze namespace health
            namespace_health = await self.health_monitor.analyze_namespace_health(namespace)

            if namespace_health["recommendation"] == "needs_pruning":
                # Sample memories from namespace for health evaluation
                sample_vector = np.random.randn(1024)
                matches = storage.query(query_vector=sample_vector, namespace=namespace, top_k=100)

                # Evaluate health of sampled memories
                to_remove = []
                for match in matches:
                    if match["id"] in storage.memories:
                        memory = storage.memories[match["id"]]
                        health = self.health_monitor.evaluate_memory_health(memory, namespace)

                        if health.recommendation == "remove":
                            to_remove.append((memory.memory_id, health))

                # Remove unhealthy memories
                for memory_id, health in to_remove[:20]:  # Limit removals per cycle
                    storage.delete(memory_id, namespace=namespace)

                    results.append(
                        UpdateResult(
                            operation="prune",
                            memory_id=memory_id,
                            success=True,
                            timestamp=datetime.now(),
                            namespace=namespace,
                            changes={
                                "health_score": health.health_score,
                                "recommendation": health.recommendation,
                            },
                        )
                    )

                logger.info(f"Pruned {len(to_remove)} memories from namespace {namespace}")

        return results


# ============= Main Update Pipeline =============


class MemppUpdatePipeline:
    """Main pipeline for updating procedural memories in Pinecone"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        build_pipeline: MemppBuildPipeline,
        retrieval_pipeline: MemppRetrievalPipeline,
        config: UpdateConfig | None = None,
        llm_client: Any | None = None,
        embedder: EmbeddingModel | None = None,
    ):
        self.storage = storage
        self.build_pipeline = build_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self.config = config or UpdateConfig()
        # Ensure embedder is always concrete for type checkers (local var)
        if embedder is None:
            embedder_final: EmbeddingModel = build_pipeline.embedder
        else:
            from typing import cast as _cast

            embedder_final = _cast(EmbeddingModel, embedder)

        # Initialize components
        self.reflection_engine = ReflectionEngine(llm_client)
        self.consolidator = PineconeMemoryConsolidator(
            self.storage, embedder_final, self.config.consolidation_threshold
        )
        self.health_monitor = PineconeMemoryHealthMonitor(self.storage, self.config)

        # Initialize executors
        self._init_executors()

        # Update tracking
        self.update_history = deque(maxlen=1000)
        self.task_counter = 0
        self.pending_updates = []

        # Continuous learning state
        self.learning_enabled = self.config.continuous_learning
        self.last_update_time = datetime.now()

        # Metrics
        if self.config.enable_metrics:
            self._init_metrics()

    def _init_executors(self):
        """Initialize Pinecone update executors"""
        self.executors = {
            UpdateStrategy.VANILLA: PineconeVanillaUpdateExecutor(),
            UpdateStrategy.VALIDATION: PineconeValidationUpdateExecutor(self.config.success_threshold),
            UpdateStrategy.NAMESPACE_MIGRATION: PineconeNamespaceMigrationExecutor(),
            UpdateStrategy.CONSOLIDATION: PineconeConsolidationUpdateExecutor(self.consolidator),
            UpdateStrategy.PRUNING: PineconePruningUpdateExecutor(self.health_monitor),
        }

    def _init_metrics(self):
        """Initialize metrics tracking"""
        self.update_counter = Counter(
            "memory_updates_total",
            "Total number of memory updates",
            ["strategy", "namespace", "success"],
        )
        self.update_duration = Histogram(
            "memory_update_duration_seconds",
            "Duration of memory update operations",
            ["strategy"],
        )
        self.memory_count_gauge = Gauge("total_memories", "Total number of stored memories")

    async def update_after_task(
        self,
        trajectory: Trajectory,
        retrieval_result: RetrievalResult | None = None,
        strategy: UpdateStrategy = UpdateStrategy.VALIDATION,
    ) -> list[UpdateResult]:
        """Update memory after completing a task"""

        self.task_counter += 1

        # Build memory from trajectory
        memory = await self.build_pipeline.build_from_trajectory(trajectory, strategy="proceduralization")

        # Track for batch update
        self.pending_updates.append({
            "trajectory": trajectory,
            "memory": memory,
            "retrieval_result": retrieval_result,
            "timestamp": datetime.now(),
        })

        # Check if batch update is needed
        if self.task_counter % self.config.update_interval == 0:
            return await self.batch_update(strategy)

        return []

    async def batch_update(self, strategy: UpdateStrategy = UpdateStrategy.VALIDATION) -> list[UpdateResult]:
        """Perform batch update of memories"""

        if not self.pending_updates:
            return []

        logger.info(f"Performing batch update with {len(self.pending_updates)} pending updates")

        start_time = datetime.now()
        all_results = []

        # Group updates by success/failure
        successful_updates = []
        failed_updates = []

        for update in self.pending_updates:
            if update["trajectory"].status == TaskStatus.SUCCESS:
                successful_updates.append(update)
            else:
                if update["retrieval_result"]:
                    failed_updates.append({
                        "memory": update["retrieval_result"].memory,
                        "trajectory": update["trajectory"],
                        "retrieval_result": update["retrieval_result"],
                    })

        # Execute updates based on strategy
        if strategy == UpdateStrategy.VANILLA:
            memories = [u["memory"] for u in self.pending_updates]
            context = {"trajectories": [u["trajectory"] for u in self.pending_updates]}
            results = await self.executors[UpdateStrategy.VANILLA].execute(self.storage, memories, context)
            all_results.extend(results)

        elif strategy == UpdateStrategy.VALIDATION:
            if successful_updates:
                memories = [u["memory"] for u in successful_updates]
                context = {"trajectories": [u["trajectory"] for u in successful_updates]}
                results = await self.executors[UpdateStrategy.VALIDATION].execute(self.storage, memories, context)
                all_results.extend(results)

        # Perform maintenance operations
        if self.config.continuous_learning:
            # Consolidation
            if self.config.auto_index_optimization:
                consolidation_results = await self.executors[UpdateStrategy.CONSOLIDATION].execute(self.storage, [], {})
                all_results.extend(consolidation_results)

            # Pruning
            pruning_results = await self.executors[UpdateStrategy.PRUNING].execute(self.storage, [], {})
            all_results.extend(pruning_results)

            # Namespace migration if enabled
            if self.config.namespace_rebalancing:
                migration_results = await self._perform_namespace_migration()
                all_results.extend(migration_results)

        # Clear pending updates
        self.pending_updates.clear()

        # Update history
        self.update_history.append({
            "timestamp": datetime.now(),
            "strategy": strategy.name,
            "results_count": len(all_results),
            "duration": (datetime.now() - start_time).total_seconds(),
        })

        # Update metrics
        if self.config.enable_metrics:
            stats = self.storage.get_statistics()
            self.memory_count_gauge.set(stats["total_vectors"])

            for result in all_results:
                self.update_counter.labels(
                    strategy=result.operation,
                    namespace=result.namespace,
                    success=str(result.success),
                ).inc()

        logger.info(f"Batch update completed: {len(all_results)} operations")
        return all_results

    async def _perform_namespace_migration(self) -> list[UpdateResult]:
        """Perform intelligent namespace migration"""

        migrations = []

        # Check trajectory memories that might be ready for proceduralization
        trajectory_matches = self.storage.query(
            query_vector=np.random.randn(1024),
            namespace="trajectory",
            top_k=20,
            filter={"success_rate": {"$gte": 0.8}, "usage_count": {"$gte": 5}},
        )

        for match in trajectory_matches[:5]:  # Limit migrations per cycle
            if match["id"] in self.storage.memories:
                memory = self.storage.memories[match["id"]]
                migrations.append({
                    "memory": memory,
                    "from_namespace": "trajectory",
                    "to_namespace": "proceduralized",
                })

        if migrations:
            return await self.executors[UpdateStrategy.NAMESPACE_MIGRATION].execute(
                self.storage, [], {"migrations": migrations}
            )

        return []

    async def continuous_learning_cycle(self):
        """Run continuous learning cycle with Pinecone optimization"""

        while self.learning_enabled:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Get Pinecone index statistics
                stats = self.storage.get_statistics()

                logger.info(
                    f"Pinecone index status: {stats['total_vectors']} vectors, "
                    f"fullness: {stats.get('index_fullness', 0):.2%}"
                )

                # Trigger maintenance if needed
                if stats["total_vectors"] > self.config.max_total_memories * 0.9:
                    logger.info("Triggering maintenance due to high memory count")

                    # Aggressive pruning
                    await self.executors[UpdateStrategy.PRUNING].execute(self.storage, [], {})

                    # Consolidation
                    await self.executors[UpdateStrategy.CONSOLIDATION].execute(self.storage, [], {})

                # Namespace rebalancing
                if self.config.namespace_rebalancing:
                    for namespace, ns_stats in stats.get("namespaces", {}).items():
                        if ns_stats.get("vector_count", 0) > self.config.max_memory_per_namespace:
                            logger.info(f"Namespace {namespace} needs rebalancing")
                            await self._perform_namespace_migration()

            except Exception as e:
                logger.error(f"Error in continuous learning cycle: {e}")
                await asyncio.sleep(60)

    def get_update_statistics(self) -> dict[str, Any]:
        """Get update pipeline statistics"""

        # Calculate update frequency
        if len(self.update_history) > 1:
            time_diffs = []
            for i in range(1, len(self.update_history)):
                diff = (self.update_history[i]["timestamp"] - self.update_history[i - 1]["timestamp"]).total_seconds()
                time_diffs.append(diff)
            avg_interval = np.mean(time_diffs)
        else:
            avg_interval = 0

        # Strategy usage
        strategy_usage = defaultdict(int)
        for update in self.update_history:
            strategy_usage[update["strategy"]] += 1

        # Get Pinecone statistics
        pinecone_stats = self.storage.get_statistics()

        return {
            "total_updates": len(self.update_history),
            "pending_updates": len(self.pending_updates),
            "task_counter": self.task_counter,
            "average_update_interval": avg_interval,
            "strategy_usage": dict(strategy_usage),
            "pinecone_stats": pinecone_stats,
            "last_update": self.update_history[-1]["timestamp"].isoformat() if self.update_history else None,
            "continuous_learning_enabled": self.learning_enabled,
        }

    def enable_continuous_learning(self):
        """Enable continuous learning"""
        self.learning_enabled = True
        asyncio.create_task(self.continuous_learning_cycle())
        logger.info("Continuous learning enabled")

    def disable_continuous_learning(self):
        """Disable continuous learning"""
        self.learning_enabled = False
        logger.info("Continuous learning disabled")


# ============= Example Usage =============


async def example_usage():
    """Example of how to use the Memp Update pipeline with Pinecone"""

    import os

    # Initialize components
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-api-key")
    storage = PineconeMemoryStorage(api_key=pinecone_api_key)

    build_pipeline = MemppBuildPipeline(pinecone_api_key=pinecone_api_key, storage=storage)
    retrieval_pipeline = MemppRetrievalPipeline(storage=storage)

    # Initialize update pipeline
    update_config = UpdateConfig(
        batch_size=5,
        success_threshold=0.7,
        reflection_enabled=True,
        continuous_learning=True,
        update_interval=3,  # Update every 3 tasks for demo
        namespace_rebalancing=True,
        auto_index_optimization=True,
    )

    update_pipeline = MemppUpdatePipeline(
        storage=storage,
        build_pipeline=build_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        config=update_config,
    )

    # Enable continuous learning
    update_pipeline.enable_continuous_learning()

    # Simulate task completions
    print("=== Simulating Task Completions with Pinecone ===\n")

    from mempp.build import (
        TaskStatus,
        Trajectory,
    )

    sample_trajectories = [
        Trajectory(
            task_id="task_001",
            task_description="Clean cup and put in microwave",
            states=[],
            actions=[],
            observations=[],
            status=TaskStatus.SUCCESS,
            final_reward=1.0,
        ),
        Trajectory(
            task_id="task_002",
            task_description="Heat egg and dispose",
            states=[],
            actions=[],
            observations=[],
            status=TaskStatus.FAILURE,
            final_reward=0.0,
        ),
        Trajectory(
            task_id="task_003",
            task_description="Organize kitchen items",
            states=[],
            actions=[],
            observations=[],
            status=TaskStatus.PARTIAL,
            final_reward=0.5,
        ),
        Trajectory(
            task_id="task_004",
            task_description="Clean plate and store",
            states=[],
            actions=[],
            observations=[],
            status=TaskStatus.SUCCESS,
            final_reward=0.9,
        ),
    ]

    # Process trajectories
    for i, trajectory in enumerate(sample_trajectories):
        print(f"Processing task {i + 1}: {trajectory.task_description}")
        print(f"  Status: {trajectory.status.name}, Reward: {trajectory.final_reward}")

        # Simulate retrieval for failed tasks
        retrieval_result = None
        if trajectory.status != TaskStatus.SUCCESS and len(storage.memories) > 0:
            memories = list(storage.memories.values())
            if memories:
                from mempp.retrieve import RetrievalResult

                retrieval_result = RetrievalResult(
                    memory=memories[0],
                    score=0.8,
                    strategy_used="query_based",
                    retrieval_time=0.1,
                    namespace="proceduralized",
                )

        # Update after task
        results = await update_pipeline.update_after_task(
            trajectory, retrieval_result, strategy=UpdateStrategy.VALIDATION
        )

        if results:
            print(f"  Batch update triggered: {len(results)} operations")
            for result in results[:3]:
                print(
                    f"    - {result.operation} in {result.namespace}: "
                    f"{result.memory_id[:8]}... (success: {result.success})"
                )

    # Get statistics
    print("\n=== Update Statistics ===")
    stats = update_pipeline.get_update_statistics()
    print(f"Total Tasks: {stats['task_counter']}")
    print(f"Pending Updates: {stats['pending_updates']}")
    print("Pinecone Statistics:")
    print(f"  Total Vectors: {stats['pinecone_stats']['total_vectors']}")
    print(f"  Namespaces: {list(stats['pinecone_stats'].get('namespaces', {}).keys())}")

    # Disable continuous learning
    update_pipeline.disable_continuous_learning()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
