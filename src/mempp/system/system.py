import asyncio
import json
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
from pinecone_text.sparse import BM25Encoder
import anthropic
from aiokafka import AIOKafkaProducer
import ray
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from fastapi import FastAPI
import uvicorn
import psutil
import yaml

# Import all components with Pinecone integration
from mempp.build import (
    ProceduralMemory,
    PineconeMemoryStorage,
    Trajectory,
    TaskStatus,
    Action,
    State,
    Observation,
    ActionType,
    MempBuildPipeline,
    MultilingualE5Embedder,
)
from mempp.retrieve import (
    MempRetrievalPipeline,
    RetrievalResult,
    RetrievalConfig,
    RetrievalStrategy,
)
from mempp.update import MempUpdatePipeline, UpdateConfig, UpdateStrategy, UpdateResult

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ============= Task Execution Framework =============


class TaskExecutor(Protocol):
    """Protocol for task execution"""

    async def execute(
        self, task_description: str, memory_context: Optional[List[RetrievalResult]] = None
    ) -> Trajectory: ...


@dataclass
class TaskRequest:
    """Request for task execution"""

    task_id: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    preferred_namespace: Optional[str] = None  # Pinecone namespace preference
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class TaskResponse:
    """Response from task execution"""

    task_id: str
    status: TaskStatus
    trajectory: Optional[Trajectory]
    retrieved_memories: List[RetrievalResult]
    built_memory: Optional[ProceduralMemory]
    update_results: List[UpdateResult]
    execution_time: float
    namespaces_used: List[str] = field(default_factory=list)  # Pinecone namespaces accessed
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============= Configuration Management =============


@dataclass
class MempSystemConfig:
    """Complete system configuration with Pinecone settings"""

    # Pinecone configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "memp-memories"
    pinecone_metric: str = "dotproduct"
    pinecone_use_serverless: bool = True
    pinecone_dimension: int = 1024

    # Namespace configuration
    default_namespaces: List[str] = field(default_factory=lambda: ["proceduralized", "script", "trajectory"])
    namespace_auto_balance: bool = True
    max_vectors_per_namespace: int = 5000

    # Build configuration
    build_strategy: str = "proceduralization"
    enable_sparse_vectors: bool = True  # Enable hybrid search

    # Retrieval configuration
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.CASCADING
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranking: bool = True
    use_caching: bool = True
    use_hybrid_search: bool = True  # Pinecone hybrid search
    alpha: float = 0.7  # Dense vs sparse weight

    # Update configuration
    update_strategy: UpdateStrategy = UpdateStrategy.ADJUSTMENT
    update_interval: int = 10
    reflection_enabled: bool = True
    continuous_learning: bool = True
    max_total_memories: int = 20000
    consolidation_threshold: float = 0.9
    namespace_migration_enabled: bool = True

    # System configuration
    enable_metrics: bool = True
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    kafka_brokers: Optional[List[str]] = None

    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Performance
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0
    batch_size: int = 20

    @classmethod
    def from_yaml(cls, path: Path) -> "MempSystemConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        config_dict = self.__dict__.copy()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# ============= Event System =============


class EventType(Enum):
    """System event types"""

    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    MEMORY_RETRIEVED = auto()
    MEMORY_BUILT = auto()
    MEMORY_UPDATED = auto()
    MEMORY_CONSOLIDATED = auto()
    MEMORY_PRUNED = auto()
    NAMESPACE_MIGRATED = auto()  # Pinecone namespace migration
    INDEX_OPTIMIZED = auto()  # Pinecone index optimization
    SYSTEM_HEALTH_CHECK = auto()


@dataclass
class SystemEvent:
    """System event"""

    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
        }


class EventBus:
    """Event bus for system-wide communication"""

    def __init__(self, kafka_brokers: Optional[List[str]] = None):
        self.listeners = defaultdict(list)
        self.kafka_brokers = kafka_brokers
        self.producer = None
        self.consumer = None

    async def initialize(self):
        """Initialize event bus"""
        if self.kafka_brokers:
            try:
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=",".join(self.kafka_brokers), value_serializer=lambda v: json.dumps(v).encode()
                )
                await self.producer.start()
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Kafka: {e}")

    async def publish(self, event: SystemEvent):
        """Publish an event"""
        # Local listeners
        for listener in self.listeners[event.event_type]:
            try:
                await listener(event)
            except Exception as e:
                logger.error(f"Listener error: {e}")

        # Kafka if available
        if self.producer:
            try:
                await self.producer.send("memp_events", value=event.to_dict())
            except Exception as e:
                logger.error(f"Kafka publish error: {e}")

    def subscribe(self, event_type: EventType, listener: Callable):
        """Subscribe to events"""
        self.listeners[event_type].append(listener)

    async def close(self):
        """Close event bus"""
        if self.producer:
            await self.producer.stop()
        if self.consumer:
            await self.consumer.stop()


# ============= Task Execution Implementations =============


class SimulatedTaskExecutor(TaskExecutor):
    """Simulated task executor for testing"""

    def __init__(self, success_rate: float = 0.8):
        self.success_rate = success_rate

    async def execute(
        self, task_description: str, memory_context: Optional[List[RetrievalResult]] = None
    ) -> Trajectory:
        """Simulate task execution"""

        await asyncio.sleep(np.random.uniform(0.5, 2.0))

        # Boost success rate if good memories are retrieved
        if memory_context and memory_context[0].score > 0.8:
            success_prob = min(0.95, self.success_rate + 0.15)
        else:
            success_prob = self.success_rate

        success = np.random.random() < success_prob

        # Generate simulated trajectory
        states = [
            State(f"Initial state for {task_description}", datetime.now()),
            State(f"Processing {task_description}", datetime.now()),
            State(f"Final state for {task_description}", datetime.now()),
        ]

        actions = [
            Action(ActionType.PLANNING, f"Plan for {task_description}", datetime.now()),
            Action(ActionType.TOOL_USE, f"Execute {task_description}", datetime.now()),
            Action(ActionType.OBSERVATION, "Check result", datetime.now()),
        ]

        observations = [
            Observation("Task initiated", datetime.now(), 0.1),
            Observation("Processing", datetime.now(), 0.3),
            Observation("Completed" if success else "Failed", datetime.now(), 1.0 if success else 0.0),
        ]

        return Trajectory(
            task_id=str(uuid.uuid4()),
            task_description=task_description,
            states=states,
            actions=actions,
            observations=observations,
            status=TaskStatus.SUCCESS if success else TaskStatus.FAILURE,
            final_reward=1.0 if success else 0.0,
            metadata={"simulated": True, "memory_aided": bool(memory_context)},
        )


class LLMTaskExecutor(TaskExecutor):
    """LLM-based task executor"""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client or anthropic.Anthropic()

    async def execute(
        self, task_description: str, memory_context: Optional[List[RetrievalResult]] = None
    ) -> Trajectory:
        """Execute task using LLM"""

        # Build context from memories
        context = ""
        if memory_context:
            context = "Previous successful approaches (with namespaces):\n"
            for i, result in enumerate(memory_context[:3]):
                context += f"{i + 1}. [{result.namespace}] {result.memory.task_pattern}\n"

        # Generate execution plan
        prompt = f"""
        Task: {task_description}
        
        {context}
        
        Generate a step-by-step execution plan and simulate the execution.
        Provide actions, observations, and final status.
        """

        response = await asyncio.to_thread(
            self.llm_client.messages.create,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_llm_response(response.content[0].text, task_description)

    def _parse_llm_response(self, response: str, task_description: str) -> Trajectory:
        """Parse LLM response into trajectory"""

        lines = response.strip().split("\n")
        actions = []
        observations = []

        for line in lines:
            if "action" in line.lower() or "step" in line.lower():
                actions.append(Action(ActionType.TOOL_USE, line.strip(), datetime.now()))
            elif "observe" in line.lower() or "result" in line.lower():
                observations.append(Observation(line.strip(), datetime.now(), 0.5))

        success = "success" in response.lower() or "completed" in response.lower()

        return Trajectory(
            task_id=str(uuid.uuid4()),
            task_description=task_description,
            states=[State("LLM execution", datetime.now())],
            actions=actions if actions else [Action(ActionType.TOOL_USE, "Execute", datetime.now())],
            observations=observations if observations else [Observation("Done", datetime.now(), 1.0)],
            status=TaskStatus.SUCCESS if success else TaskStatus.FAILURE,
            final_reward=1.0 if success else 0.0,
            metadata={"executor": "llm"},
        )


# ============= Core MempSystem with Pinecone =============


class MempSystem:
    """Core system integrating all Memp components with Pinecone"""

    def __init__(self, config: MempSystemConfig, task_executor: Optional[TaskExecutor] = None):
        self.config = config
        self.task_executor = task_executor or SimulatedTaskExecutor()

        # Initialize Pinecone storage
        self.storage = PineconeMemoryStorage(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
            index_name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            use_serverless=config.pinecone_use_serverless,
        )

        # Initialize embedder and sparse encoder
        self.embedder = MultilingualE5Embedder()
        self.sparse_encoder = BM25Encoder() if config.enable_sparse_vectors else None

        # Initialize pipelines
        self._init_pipelines()

        # Initialize event system
        self.event_bus = EventBus(self.config.kafka_brokers)

        # Task management
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.active_tasks = {}
        self.task_history = deque(maxlen=10000)

        # Metrics
        if self.config.enable_metrics:
            self._init_metrics()

        # State
        self.is_running = False
        self.workers = []

        # Statistics
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_memories": 0,
            "namespace_distribution": defaultdict(int),
            "average_retrieval_score": 0,
            "average_execution_time": 0,
        }

        logger.info(
            "MempSystem initialized with Pinecone",
            index=config.pinecone_index_name,
            environment=config.pinecone_environment,
        )

    def _init_pipelines(self):
        """Initialize component pipelines with Pinecone"""

        # Build pipeline
        self.build_pipeline = MempBuildPipeline(
            pinecone_api_key=self.config.pinecone_api_key,
            embedder=self.embedder,
            storage=self.storage,
            llm_client=anthropic.Anthropic() if self.config.anthropic_api_key else None,
        )

        # Retrieval pipeline
        retrieval_config = RetrievalConfig(
            strategy=self.config.retrieval_strategy,
            top_k=self.config.retrieval_top_k,
            similarity_threshold=self.config.similarity_threshold,
            use_reranking=self.config.use_reranking,
            use_caching=self.config.use_caching,
            use_hybrid_search=self.config.use_hybrid_search,
            search_namespaces=self.config.default_namespaces,
            alpha=self.config.alpha,
        )

        self.retrieval_pipeline = MempRetrievalPipeline(
            storage=self.storage,
            config=retrieval_config,
            embedder=self.embedder,
            cache_redis_url=self.config.redis_url,
            gemini_api_key=self.config.gemini_api_key,
        )

        # Update pipeline
        update_config = UpdateConfig(
            batch_size=self.config.batch_size,
            reflection_enabled=self.config.reflection_enabled,
            continuous_learning=self.config.continuous_learning,
            update_interval=self.config.update_interval,
            max_memory_per_namespace=self.config.max_vectors_per_namespace,
            max_total_memories=self.config.max_total_memories,
            consolidation_threshold=self.config.consolidation_threshold,
            namespace_rebalancing=self.config.namespace_migration_enabled,
            enable_metrics=self.config.enable_metrics,
        )

        self.update_pipeline = MempUpdatePipeline(
            storage=self.storage,
            build_pipeline=self.build_pipeline,
            retrieval_pipeline=self.retrieval_pipeline,
            config=update_config,
            llm_client=anthropic.Anthropic() if self.config.anthropic_api_key else None,
            embedder=self.embedder,
        )

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        self.task_counter = Counter("memp_tasks_total", "Total number of tasks processed", ["status", "namespace"])
        self.memory_counter = Gauge("memp_memories_total", "Total number of stored memories")
        self.namespace_counter = Gauge("memp_namespace_vectors", "Vectors per namespace", ["namespace"])
        self.execution_time = Histogram("memp_task_execution_seconds", "Task execution time in seconds")
        self.retrieval_score = Summary("memp_retrieval_score", "Memory retrieval relevance scores")

    async def initialize(self):
        """Initialize the system"""
        await self.event_bus.initialize()

        # Train sparse encoder if enabled
        if self.sparse_encoder:
            sample_texts = ["clean cup microwave kitchen", "heat egg dispose garbage", "organize items shelf storage"]
            self.sparse_encoder.fit(sample_texts)

        if self.config.continuous_learning:
            self.update_pipeline.enable_continuous_learning()

        if self.config.enable_distributed:
            await self._init_distributed()

        logger.info("MempSystem initialized successfully")

    async def _init_distributed(self):
        """Initialize distributed components"""
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
            logger.info("Ray cluster initialized")

    # ============= Main Pipeline with Pinecone =============

    async def process_task(self, request: TaskRequest) -> TaskResponse:
        """Main pipeline: Retrieve → Execute → Build → Update using Pinecone"""

        start_time = datetime.now()
        namespaces_used = []

        try:
            # Emit start event
            await self.event_bus.publish(
                SystemEvent(
                    EventType.TASK_STARTED,
                    datetime.now(),
                    {"task_id": request.task_id, "description": request.description},
                    "MempSystem",
                )
            )

            # Step 1: Retrieve relevant memories from Pinecone
            logger.info("Retrieving memories from Pinecone", task_id=request.task_id)

            # Use preferred namespace if specified
            if request.preferred_namespace:
                original_namespaces = self.retrieval_pipeline.config.search_namespaces
                self.retrieval_pipeline.config.search_namespaces = [request.preferred_namespace]

            retrieved_memories = await self.retrieval_pipeline.retrieve(
                query=request.description,
                strategy=self.config.retrieval_strategy,
                top_k=self.config.retrieval_top_k,
                namespace=request.preferred_namespace,
                use_external_search=bool(self.config.gemini_api_key),
            )

            # Restore original namespaces
            if request.preferred_namespace:
                self.retrieval_pipeline.config.search_namespaces = original_namespaces

            # Track namespaces used
            namespaces_used = list(set([r.namespace for r in retrieved_memories]))

            # Log retrieval scores
            if retrieved_memories:
                avg_score = np.mean([r.score for r in retrieved_memories])
                if self.config.enable_metrics:
                    self.retrieval_score.observe(avg_score)
                logger.info(
                    f"Retrieved {len(retrieved_memories)} memories", avg_score=avg_score, namespaces=namespaces_used
                )

            # Step 2: Execute task with memory context
            logger.info("Executing task", task_id=request.task_id)
            trajectory = await self.task_executor.execute(
                task_description=request.description, memory_context=retrieved_memories
            )

            # Step 3: Build new memory from execution
            logger.info("Building memory", task_id=request.task_id)
            built_memory = await self.build_pipeline.build_from_trajectory(
                trajectory=trajectory, strategy=self.config.build_strategy
            )

            # Step 4: Update memory system in Pinecone
            logger.info("Updating Pinecone memory system", task_id=request.task_id)

            # Determine which memory led to failure (if any)
            retrieval_result = retrieved_memories[0] if retrieved_memories else None

            update_results = await self.update_pipeline.update_after_task(
                trajectory=trajectory,
                retrieval_result=retrieval_result if trajectory.status != TaskStatus.SUCCESS else None,
                strategy=self.config.update_strategy,
            )

            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(trajectory, execution_time, namespaces_used)

            # Update metrics
            if self.config.enable_metrics:
                for namespace in namespaces_used:
                    self.task_counter.labels(status=trajectory.status.name, namespace=namespace).inc()
                self.execution_time.observe(execution_time)

                # Update namespace metrics
                stats = self.storage.get_statistics()
                self.memory_counter.set(stats["total_vectors"])
                for ns, ns_stats in stats.get("namespaces", {}).items():
                    self.namespace_counter.labels(namespace=ns).set(ns_stats.get("vector_count", 0))

            # Create response
            response = TaskResponse(
                task_id=request.task_id,
                status=trajectory.status,
                trajectory=trajectory,
                retrieved_memories=retrieved_memories,
                built_memory=built_memory,
                update_results=update_results,
                execution_time=execution_time,
                namespaces_used=namespaces_used,
                metadata={
                    "memory_aided": bool(retrieved_memories),
                    "avg_retrieval_score": np.mean([r.score for r in retrieved_memories]) if retrieved_memories else 0,
                    "pinecone_index": self.config.pinecone_index_name,
                },
            )

            # Emit completion event
            await self.event_bus.publish(
                SystemEvent(
                    EventType.TASK_COMPLETED,
                    datetime.now(),
                    {"task_id": request.task_id, "status": trajectory.status.name, "namespaces": namespaces_used},
                    "MempSystem",
                )
            )

            # Store in history
            self.task_history.append(response)

            logger.info(
                "Task completed successfully",
                task_id=request.task_id,
                status=trajectory.status.name,
                execution_time=execution_time,
                namespaces=namespaces_used,
            )

            return response

        except Exception as e:
            logger.error("Task processing failed", task_id=request.task_id, error=str(e))

            # Emit failure event
            await self.event_bus.publish(
                SystemEvent(
                    EventType.TASK_FAILED, datetime.now(), {"task_id": request.task_id, "error": str(e)}, "MempSystem"
                )
            )

            # Return error response
            return TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILURE,
                trajectory=None,
                retrieved_memories=[],
                built_memory=None,
                update_results=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                namespaces_used=[],
                metadata={"error": str(e)},
            )

    def _update_statistics(self, trajectory: Trajectory, execution_time: float, namespaces: List[str]):
        """Update system statistics"""
        self.stats["total_tasks"] += 1

        if trajectory.status == TaskStatus.SUCCESS:
            self.stats["successful_tasks"] += 1
        else:
            self.stats["failed_tasks"] += 1

        # Update namespace distribution
        for namespace in namespaces:
            self.stats["namespace_distribution"][namespace] += 1

        # Get current Pinecone stats
        pinecone_stats = self.storage.get_statistics()
        self.stats["total_memories"] = pinecone_stats["total_vectors"]

        # Update moving averages
        alpha = 0.1  # Exponential moving average factor
        self.stats["average_execution_time"] = (
            alpha * execution_time + (1 - alpha) * self.stats["average_execution_time"]
        )

    # ============= Pinecone-specific Operations =============

    async def optimize_pinecone_index(self):
        """Optimize Pinecone index for better performance"""
        logger.info("Starting Pinecone index optimization")

        # Get current statistics
        stats = self.storage.get_statistics()

        # Perform namespace-specific optimizations
        for namespace in self.config.default_namespaces:
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            vector_count = ns_stats.get("vector_count", 0)

            if vector_count > self.config.max_vectors_per_namespace * 0.9:
                logger.info(f"Namespace {namespace} needs optimization: {vector_count} vectors")

                # Trigger consolidation for this namespace
                await self.update_pipeline.executors[UpdateStrategy.CONSOLIDATION].execute(
                    self.storage, [], {"target_namespace": namespace}
                )

        # Emit optimization event
        await self.event_bus.publish(
            SystemEvent(
                EventType.INDEX_OPTIMIZED,
                datetime.now(),
                {"index": self.config.pinecone_index_name, "stats": stats},
                "MempSystem",
            )
        )

        logger.info("Pinecone index optimization completed")

    async def migrate_namespace(self, from_namespace: str, to_namespace: str, criteria: Optional[Dict] = None):
        """Migrate memories between Pinecone namespaces"""
        logger.info(f"Migrating from {from_namespace} to {to_namespace}")

        # Query memories from source namespace
        sample_vector = np.random.randn(self.config.pinecone_dimension)
        matches = self.storage.query(query_vector=sample_vector, namespace=from_namespace, top_k=100, filter=criteria)

        migrations = []
        for match in matches:
            if match["id"] in self.storage.memories:
                memory = self.storage.memories[match["id"]]
                migrations.append({"memory": memory, "from_namespace": from_namespace, "to_namespace": to_namespace})

        # Execute migrations
        if migrations:
            from mempp.update import PineconeNamespaceMigrationExecutor

            executor = PineconeNamespaceMigrationExecutor()
            results = await executor.execute(self.storage, [], {"migrations": migrations})

            # Emit migration event
            await self.event_bus.publish(
                SystemEvent(
                    EventType.NAMESPACE_MIGRATED,
                    datetime.now(),
                    {"from": from_namespace, "to": to_namespace, "count": len(results), "criteria": criteria},
                    "MempSystem",
                )
            )

            logger.info(f"Migrated {len(results)} memories")
            return results

        return []

    # ============= System Management =============

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check including Pinecone status"""

        health = {"status": "healthy", "timestamp": datetime.now().isoformat(), "components": {}, "metrics": {}}

        # Check Pinecone storage
        try:
            pinecone_stats = self.storage.get_statistics()
            health["components"]["pinecone"] = {
                "status": "healthy",
                "index": self.config.pinecone_index_name,
                "total_vectors": pinecone_stats["total_vectors"],
                "namespaces": list(pinecone_stats.get("namespaces", {}).keys()),
                "index_fullness": pinecone_stats.get("index_fullness", 0),
            }
        except Exception as e:
            health["components"]["pinecone"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"

        # Check pipelines
        health["components"]["build_pipeline"] = {
            "status": "healthy",
            "stats": self.build_pipeline.get_build_statistics(),
        }

        health["components"]["retrieval_pipeline"] = {
            "status": "healthy",
            "stats": self.retrieval_pipeline.get_retrieval_statistics(),
        }

        health["components"]["update_pipeline"] = {
            "status": "healthy",
            "stats": self.update_pipeline.get_update_statistics(),
        }

        # System metrics
        health["metrics"] = {
            "total_tasks": self.stats["total_tasks"],
            "success_rate": (self.stats["successful_tasks"] / max(1, self.stats["total_tasks"])),
            "namespace_distribution": dict(self.stats["namespace_distribution"]),
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        }

        # Emit health check event
        await self.event_bus.publish(SystemEvent(EventType.SYSTEM_HEALTH_CHECK, datetime.now(), health, "MempSystem"))

        return health

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""

        pinecone_stats = self.storage.get_statistics()

        return {
            "system": self.stats,
            "pinecone": pinecone_stats,
            "build": self.build_pipeline.get_build_statistics(),
            "retrieval": self.retrieval_pipeline.get_retrieval_statistics(),
            "update": self.update_pipeline.get_update_statistics(),
            "queue": {"size": self.task_queue.qsize(), "active_tasks": len(self.active_tasks)},
        }

    async def shutdown(self):
        """Gracefully shutdown the system"""

        logger.info("Shutting down MempSystem")

        # Stop workers
        if self.workers:
            self.is_running = False
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()

        # Final batch update
        if self.update_pipeline.pending_updates:
            await self.update_pipeline.batch_update(self.config.update_strategy)

        # Optimize Pinecone index before shutdown
        await self.optimize_pinecone_index()

        # Disable continuous learning
        if self.config.continuous_learning:
            self.update_pipeline.disable_continuous_learning()

        # Close event bus
        await self.event_bus.close()

        # Export final statistics
        stats = self.get_statistics()
        logger.info("Final statistics", stats=stats)

        logger.info("MempSystem shutdown complete")


# ============= REST API =============


class MempSystemAPI:
    """FastAPI wrapper for MempSystem with Pinecone endpoints"""

    def __init__(self, memp_system: MempSystem):
        self.system = memp_system
        self.app = FastAPI(title="MempSystem API with Pinecone", version="2.0.0")
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.post("/task")
        async def submit_task(request: TaskRequest) -> Dict[str, str]:
            """Submit a task for processing"""
            task_id = await self.system.submit_task(request)
            return {"task_id": task_id, "status": "submitted"}

        @self.app.get("/task/{task_id}")
        async def get_task_result(task_id: str) -> Dict[str, Any]:
            """Get task result"""
            result = await self.system.get_task_result(task_id)
            if result:
                return {
                    "task_id": result.task_id,
                    "status": result.status.name,
                    "execution_time": result.execution_time,
                    "namespaces_used": result.namespaces_used,
                    "metadata": result.metadata,
                }
            return {"error": "Task not found or still processing"}

        @self.app.post("/migrate")
        async def migrate_namespace(from_ns: str, to_ns: str, criteria: Optional[Dict] = None) -> Dict[str, Any]:
            """Migrate memories between Pinecone namespaces"""
            results = await self.system.migrate_namespace(from_ns, to_ns, criteria)
            return {"migrated": len(results), "from": from_ns, "to": to_ns}

        @self.app.post("/optimize")
        async def optimize_index() -> Dict[str, str]:
            """Optimize Pinecone index"""
            await self.system.optimize_pinecone_index()
            return {"status": "optimization completed"}

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """System health check"""
            return await self.system.health_check()

        @self.app.get("/stats")
        async def get_statistics() -> Dict[str, Any]:
            """Get system statistics"""
            return self.system.get_statistics()

        @self.app.on_event("startup")
        async def startup():
            """Initialize system on startup"""
            await self.system.initialize()

        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown"""
            await self.system.shutdown()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)


# ============= Example Usage =============


async def example_usage():
    """Comprehensive example of MempSystem with Pinecone"""

    import os

    # Create configuration
    config = MempSystemConfig(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "your-api-key"),
        pinecone_environment="us-east-1",
        pinecone_index_name="memp-demo",
        retrieval_strategy=RetrievalStrategy.CASCADING,
        update_strategy=UpdateStrategy.ADJUSTMENT,
        continuous_learning=True,
        namespace_migration_enabled=True,
        enable_metrics=True,
    )

    # Initialize system
    system = MempSystem(config=config)
    await system.initialize()

    print("=== MempSystem with Pinecone Demo ===\n")

    # Example tasks
    task_descriptions = [
        "Clean the cup and put it in the microwave",
        "Heat an egg and dispose of it",
        "Organize kitchen utensils on the shelf",
        "Clean the plate and store it properly",
        "Prepare coffee and serve it hot",
    ]

    print("=== Processing Tasks with Namespace Awareness ===")

    # Process tasks with different namespace preferences
    for i, desc in enumerate(task_descriptions[:3]):
        # Alternate namespace preferences
        namespace = ["proceduralized", "script", "trajectory"][i % 3]

        request = TaskRequest(task_id=f"demo_{i}", description=desc, preferred_namespace=namespace)

        print(f"\nTask {i + 1}: {desc}")
        print(f"  Preferred Namespace: {namespace}")
        response = await system.process_task(request)

        print(f"  Status: {response.status.name}")
        print(f"  Execution Time: {response.execution_time:.2f}s")
        print(f"  Namespaces Used: {response.namespaces_used}")
        print(f"  Retrieved Memories: {len(response.retrieved_memories)}")
        if response.retrieved_memories:
            print(f"  Best Match Score: {response.retrieved_memories[0].score:.3f}")

    # Check Pinecone statistics
    print("\n=== Pinecone Statistics ===")
    stats = system.get_statistics()
    pinecone_stats = stats["pinecone"]
    print(f"Total Vectors: {pinecone_stats['total_vectors']}")
    print(f"Index Fullness: {pinecone_stats.get('index_fullness', 0):.2%}")
    print(f"Namespaces: {list(pinecone_stats.get('namespaces', {}).keys())}")

    for ns, ns_stats in pinecone_stats.get("namespaces", {}).items():
        print(f"  {ns}: {ns_stats.get('vector_count', 0)} vectors")

    # Test namespace migration
    print("\n=== Testing Namespace Migration ===")
    migration_results = await system.migrate_namespace(
        from_namespace="trajectory", to_namespace="proceduralized", criteria={"success_rate": {"$gte": 0.8}}
    )
    print(f"Migrated {len(migration_results)} high-performing memories")

    # Optimize Pinecone index
    print("\n=== Optimizing Pinecone Index ===")
    await system.optimize_pinecone_index()
    print("Index optimization completed")

    # System health check
    print("\n=== System Health Check ===")
    health = await system.health_check()
    print(f"Status: {health['status']}")
    print(f"Pinecone Status: {health['components']['pinecone']['status']}")
    print(f"Success Rate: {health['metrics']['success_rate']:.2%}")

    # Cleanup
    await system.shutdown()
    print("\n=== Demo Complete ===")


# ============= CLI Interface =============


async def main():
    """Main entry point with CLI interface"""

    import argparse
    import os

    parser = argparse.ArgumentParser(description="MempSystem with Pinecone - Agent Procedural Memory Framework")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=["demo", "api", "worker"], default="demo", help="Execution mode")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--pinecone-key", type=str, help="Pinecone API key")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = MempSystemConfig.from_yaml(Path(args.config))
    else:
        # Use environment variable or command line argument for Pinecone key
        pinecone_key = args.pinecone_key or os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            print("Error: Pinecone API key required. Set PINECONE_API_KEY or use --pinecone-key")
            return

        config = MempSystemConfig(pinecone_api_key=pinecone_key)

    # Initialize system
    system = MempSystem(config=config)

    if args.mode == "demo":
        # Run demo
        await example_usage()

    elif args.mode == "api":
        # Start API server
        api = MempSystemAPI(system)
        api.run(port=args.port)

    elif args.mode == "worker":
        # Run as worker
        await system.initialize()

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                health = await system.health_check()
                logger.info("Health check", status=health["status"])
        except KeyboardInterrupt:
            await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
