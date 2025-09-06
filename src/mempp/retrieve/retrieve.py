import asyncio
import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, cast

import google.genai as genai
import numpy as np
import redis
import spacy
import yake
from cachetools import TTLCache
from keybert import KeyBERT
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import pipeline

# Import from Build component
from mempp.build import (
    EmbeddingModel,
    MultilingualE5Embedder,
    PineconeMemoryStorage,
    ProceduralMemory,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============= Retrieval Strategies =============


class RetrievalStrategy(Enum):
    """Different retrieval strategies"""

    RANDOM_SAMPLE = auto()
    QUERY_BASED = auto()
    AVE_FACT = auto()
    HYBRID = auto()
    RERANK = auto()
    CASCADING = auto()
    NAMESPACE_AWARE = auto()  # New: Pinecone namespace-aware strategy


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranking: bool = True
    use_caching: bool = True
    cache_ttl: int = 3600  # seconds
    enable_fallback: bool = True
    max_candidates: int = 20
    search_namespaces: list[str] = field(default_factory=lambda: ["proceduralized", "script", "trajectory"])
    use_hybrid_search: bool = True  # Enable Pinecone hybrid search
    alpha: float = 0.7  # Weight for dense vs sparse in hybrid search (0=sparse only, 1=dense only)


@dataclass
class RetrievalResult:
    """Result of memory retrieval"""

    memory: ProceduralMemory
    score: float
    strategy_used: str
    retrieval_time: float
    namespace: str = "proceduralized"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return self.score > other.score  # Higher score is better


# ============= Key Extractors =============


class KeyExtractor(ABC):
    """Abstract base for key extraction"""

    @abstractmethod
    def extract(self, text: str) -> str | list[str] | dict[str, list[str]]:
        """Extract key(s) from text"""
        pass


class QueryKeyExtractor(KeyExtractor):
    """Uses the full query as key"""

    def extract(self, text: str) -> str:
        """Return the full query"""
        return text


class KeywordExtractor(KeyExtractor):
    """Extracts keywords using multiple methods"""

    def __init__(self):
        # Initialize KeyBERT for keyword extraction
        self.keybert = KeyBERT("paraphrase-MiniLM-L6-v2")

        # Initialize YAKE for statistical keyword extraction
        self.yake = yake.KeywordExtractor(
            lan="en",
            n=3,  # max n-gram size
            dedupLim=0.7,
            top=10,
            features=None,
        )

        # Load spaCy for NER and POS tagging
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
            self.nlp = spacy.load("en_core_web_lg")

    def extract(self, text: str) -> list[str]:
        """Extract keywords using multiple methods"""
        keywords = set()

        # KeyBERT extraction
        keybert_keywords = self.keybert.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
        )
        keywords.update([kw[0] for kw in keybert_keywords])

        # YAKE extraction
        yake_keywords = self.yake.extract_keywords(text)
        keywords.update([kw[0] for kw in yake_keywords[:5]])

        # SpaCy NER and important nouns
        doc = self.nlp(text)
        # Named entities
        keywords.update([ent.text.lower() for ent in doc.ents])
        # Important nouns and verbs
        keywords.update(
            [token.text.lower() for token in doc if token.pos_ in ["NOUN", "VERB"] and not token.is_stop][:5]
        )

        return list(keywords)


class AveFactExtractor(KeyExtractor):
    """Extracts facts and averages their importance"""

    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        # Initialize fact extraction pipeline
        self.fact_extractor = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )

    def extract(self, text: str) -> dict[str, list[str]]:
        """Extract facts with importance scores"""
        # Extract keywords
        keywords = self.keyword_extractor.extract(text)

        # Extract entities as facts
        from typing import Any, cast

        entities_raw = self.fact_extractor(text)
        entities = cast(list[dict[str, Any]], entities_raw)
        facts: list[str] = [str(ent.get("word", "")) for ent in entities if float(ent.get("score", 0.0)) > 0.8]

        # Combine and weight
        all_terms = keywords + facts

        # Count frequency as importance
        term_importance = defaultdict(float)
        for term in all_terms:
            term_importance[term.lower()] += 1.0

        # We intentionally return only lists here to keep the return
        # shape consistent with the type signature and downstream usage.
        return {"keywords": keywords, "facts": facts}


# ============= Similarity Calculators =============


class SimilarityCalculator:
    """Calculates similarity between queries and memories"""

    def __init__(self, embedder: EmbeddingModel, sparse_encoder: BM25Encoder | None = None):
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        # Initialize cross-encoder for reranking
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def cosine_similarity(self, query_embedding: np.ndarray, memory_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if memory_embeddings.ndim == 1:
            memory_embeddings = memory_embeddings.reshape(1, -1)

        return cosine_similarity(query_embedding, memory_embeddings)[0]

    def cross_encode_similarity(self, query: str, memory_patterns: list[str]) -> np.ndarray:
        """Calculate similarity using cross-encoder"""
        pairs = [[query, pattern] for pattern in memory_patterns]
        scores = self.cross_encoder.predict(pairs)
        return np.array(scores)

    def create_sparse_vector(self, text: str) -> dict[str, float]:
        """Create sparse vector for hybrid search.

        Normalizes different return shapes from ``BM25Encoder`` to a
        token->weight mapping.
        """
        if not self.sparse_encoder:
            return {}

        raw: Any = self.sparse_encoder.encode_documents([text])
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return cast(dict[str, float], raw[0])
        if hasattr(raw, "indices") and hasattr(raw, "values"):
            try:
                return {str(i): float(v) for i, v in zip(raw.indices, raw.values)}  # type: ignore[attr-defined]
            except Exception:
                return {}
        return {}


# ============= Retrieval Cache =============


class RetrievalCache:
    """Caching layer for retrieval results"""

    def __init__(self, redis_url: str | None = None, ttl: int = 3600):
        self.ttl = ttl

        # Try to connect to Redis if available
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except:
                logger.warning("Redis not available, using in-memory cache")

        # Fallback to in-memory cache
        self.memory_cache = TTLCache(maxsize=1000, ttl=ttl)

    def _get_cache_key(self, query: str, strategy: str, top_k: int, namespace: str = "") -> str:
        """Generate cache key"""
        content = f"{query}_{strategy}_{top_k}_{namespace}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, query: str, strategy: str, top_k: int, namespace: str = "") -> list[RetrievalResult] | None:
        """Get cached results"""
        key = self._get_cache_key(query, strategy, top_k, namespace)

        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    from typing import cast

                    return pickle.loads(cast(bytes, cached))
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        # Try memory cache
        return self.memory_cache.get(key)

    async def set(
        self,
        query: str,
        strategy: str,
        top_k: int,
        results: list[RetrievalResult],
        namespace: str = "",
    ):
        """Cache results"""
        key = self._get_cache_key(query, strategy, top_k, namespace)

        # Cache in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, pickle.dumps(results))
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        # Always cache in memory
        self.memory_cache[key] = results


# ============= External Search Integration =============


class GeminiSearchIntegration:
    """Integration with Gemini for Google Search"""

    def __init__(self, api_key: str | None = None):
        # The google-genai library API differs from google-generativeai.
        # Use typing casts to avoid strict attribute checks.
        from typing import Any, cast as _cast

        _genai = _cast(Any, genai)
        if api_key:
            # For google-generativeai compatibility; no-op on google-genai.
            try:
                _genai.configure(api_key=api_key)
            except Exception:
                pass
        try:
            self.model = _genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            # Fallback: keep a reference to the module and call via client later.
            self.model = _genai

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_similar_tasks(self, query: str) -> list[dict[str, Any]]:
        """Search for similar tasks using Gemini with Google Search"""

        prompt = f"""
        Search for similar procedural tasks or solutions related to: {query}
        
        Focus on:
        1. Step-by-step procedures
        2. Common patterns
        3. Best practices
        4. Similar problem solutions
        
        Return structured information about similar tasks.
        """

        try:
            response = await asyncio.to_thread(
                getattr(self.model, "generate_content"), prompt, tools="google_search_retrieval"
            )

            # Parse response for similar patterns (support both .text and .output_text)
            text = getattr(response, "text", None) or getattr(response, "output_text", "")
            similar_patterns = self._parse_search_results(str(text))
            return similar_patterns

        except Exception as e:
            logger.error(f"Gemini search error: {e}")
            return []

    def _parse_search_results(self, text: str) -> list[dict[str, Any]]:
        """Parse search results into structured format"""
        patterns = []

        lines = text.strip().split("\n")
        current_pattern = {}

        for line in lines:
            if line.startswith("Task:") or line.startswith("Pattern:"):
                if current_pattern:
                    patterns.append(current_pattern)
                current_pattern = {"description": line, "steps": []}
            elif line.startswith("Step") or line.startswith("-"):
                if current_pattern:
                    current_pattern["steps"].append(line.strip("- "))

        if current_pattern:
            patterns.append(current_pattern)

        return patterns


# ============= Pinecone-based Retrievers =============


class PineconeRetriever(ABC):
    """Abstract base for Pinecone-based retrievers"""

    def __init__(self, storage: PineconeMemoryStorage, config: RetrievalConfig):
        self.storage = storage
        self.config = config

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve relevant memories"""
        pass

    def _convert_matches_to_results(
        self, matches: list, strategy: str, retrieval_time: float, namespace: str
    ) -> list[RetrievalResult]:
        """Convert Pinecone matches to RetrievalResult objects"""
        results = []

        for match in matches:
            # Retrieve full memory object from local storage
            memory = self.storage.memories.get(match["id"])

            if memory and match["score"] >= self.config.similarity_threshold:
                results.append(
                    RetrievalResult(
                        memory=memory,
                        score=match["score"],
                        strategy_used=strategy,
                        retrieval_time=retrieval_time,
                        namespace=namespace,
                        metadata=match.get("metadata", {}),
                    )
                )

        return results


class QueryBasedPineconeRetriever(PineconeRetriever):
    """Query similarity-based retriever using Pinecone"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        config: RetrievalConfig,
        embedder: EmbeddingModel,
        similarity_calc: SimilarityCalculator,
    ):
        super().__init__(storage, config)
        self.embedder = embedder
        self.similarity_calc = similarity_calc

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve based on query similarity using Pinecone"""
        start_time = datetime.now()

        # Get query embedding
        query_embedding = self.embedder.encode(query)[0]

        # Create sparse vector if hybrid search is enabled
        sparse_vector = None
        if self.config.use_hybrid_search:
            sparse_vector = self.similarity_calc.create_sparse_vector(query)

        all_results = []

        # Search across configured namespaces
        for namespace in self.config.search_namespaces:
            matches = self.storage.query(
                query_vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                sparse_vector=sparse_vector,
            )

            results = self._convert_matches_to_results(
                matches,
                "query_based_pinecone",
                (datetime.now() - start_time).total_seconds(),
                namespace,
            )

            all_results.extend(results)

        # Sort by score and return top-k
        all_results.sort()
        return all_results[:top_k]


class HybridPineconeRetriever(PineconeRetriever):
    """Hybrid retriever using Pinecone's native hybrid search"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        config: RetrievalConfig,
        embedder: EmbeddingModel,
        similarity_calc: SimilarityCalculator,
    ):
        super().__init__(storage, config)
        self.embedder = embedder
        self.similarity_calc = similarity_calc
        self.keyword_extractor = KeywordExtractor()

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve using Pinecone's hybrid search"""
        start_time = datetime.now()

        # Get dense embedding
        query_embedding = self.embedder.encode(query)[0]

        # Get sparse embedding for hybrid search
        sparse_vector = self.similarity_calc.create_sparse_vector(query)

        # Extract keywords for metadata filtering
        keywords = self.keyword_extractor.extract(query)

        all_results = []

        # Search with different strategies across namespaces
        for namespace in self.config.search_namespaces:
            # Standard hybrid search
            matches = self.storage.query(
                query_vector=query_embedding,
                namespace=namespace,
                top_k=top_k * 2,  # Get more candidates
                sparse_vector=sparse_vector,
            )

            results = self._convert_matches_to_results(
                matches,
                "hybrid_pinecone",
                (datetime.now() - start_time).total_seconds(),
                namespace,
            )

            all_results.extend(results)

        # Deduplicate and sort
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.memory.memory_id not in seen_ids:
                seen_ids.add(result.memory.memory_id)
                unique_results.append(result)

        unique_results.sort()
        return unique_results[:top_k]


class NamespaceAwareRetriever(PineconeRetriever):
    """Retriever that intelligently selects namespaces based on query"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        config: RetrievalConfig,
        embedder: EmbeddingModel,
        similarity_calc: SimilarityCalculator,
    ):
        super().__init__(storage, config)
        self.embedder = embedder
        self.similarity_calc = similarity_calc
        self.hybrid_retriever = HybridPineconeRetriever(storage, config, embedder, similarity_calc)

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Intelligently retrieve from appropriate namespaces"""
        start_time = datetime.now()

        # Determine best namespaces based on query characteristics
        namespaces = self._select_namespaces(query)

        # Temporarily update config
        original_namespaces = self.config.search_namespaces
        self.config.search_namespaces = namespaces

        # Use hybrid retriever with selected namespaces
        results = await self.hybrid_retriever.retrieve(query, top_k)

        # Restore original config
        self.config.search_namespaces = original_namespaces

        # Update strategy name
        for result in results:
            result.strategy_used = "namespace_aware"
            result.metadata["selected_namespaces"] = namespaces

        return results

    def _select_namespaces(self, query: str) -> list[str]:
        """Select appropriate namespaces based on query"""
        namespaces = []

        query_lower = query.lower()

        # Heuristics for namespace selection
        if any(word in query_lower for word in ["step", "procedure", "how to", "script"]):
            namespaces.append("script")

        if any(word in query_lower for word in ["exact", "previous", "same", "trajectory"]):
            namespaces.append("trajectory")

        # Default to proceduralized if no specific indicators
        if not namespaces or "general" in query_lower:
            namespaces.append("proceduralized")

        return namespaces


class CascadingPineconeRetriever(PineconeRetriever):
    """Cascading retriever with multiple stages using Pinecone"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        config: RetrievalConfig,
        embedder: EmbeddingModel,
        similarity_calc: SimilarityCalculator,
    ):
        super().__init__(storage, config)
        self.embedder = embedder
        self.similarity_calc = similarity_calc
        self.hybrid_retriever = HybridPineconeRetriever(storage, config, embedder, similarity_calc)
        self.namespace_retriever = NamespaceAwareRetriever(storage, config, embedder, similarity_calc)

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Multi-stage cascading retrieval"""
        start_time = datetime.now()

        # Stage 1: Get initial candidates from namespace-aware retrieval
        candidates = await self.namespace_retriever.retrieve(query, self.config.max_candidates)

        if not candidates:
            # Fallback to hybrid retrieval
            candidates = await self.hybrid_retriever.retrieve(query, self.config.max_candidates)

        if not candidates:
            return []

        # Stage 2: Rerank if enabled
        if self.config.use_reranking and len(candidates) > top_k:
            candidates = await self._rerank(query, candidates)

        # Stage 3: Apply metadata filters
        filtered = self._apply_filters(candidates)

        # Update metadata
        for result in filtered[:top_k]:
            result.strategy_used = "cascading_pinecone"
            result.retrieval_time = (datetime.now() - start_time).total_seconds()
            result.metadata["stages"] = ["namespace_aware", "rerank", "filter"]

        return filtered[:top_k]

    async def _rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder"""

        # Get memory patterns for reranking
        memory_patterns = [r.memory.task_pattern for r in candidates]

        # Calculate cross-encoder scores
        cross_scores = self.similarity_calc.cross_encode_similarity(query, memory_patterns)

        # Update scores with weighted combination
        for i, result in enumerate(candidates):
            # Combine original score with cross-encoder score
            result.score = 0.6 * result.score + 0.4 * float(cross_scores[i])

        candidates.sort()
        return candidates

    def _apply_filters(self, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Apply additional filters based on metadata"""
        filtered = []

        for result in candidates:
            # Filter by success rate
            if result.memory.success_rate >= 0.5:  # Minimum 50% success rate
                filtered.append(result)

        return filtered


# ============= Main Retrieval Pipeline =============


class MemppRetrievalPipeline:
    """Main pipeline for retrieving procedural memories using Pinecone"""

    def __init__(
        self,
        storage: PineconeMemoryStorage,
        config: RetrievalConfig | None = None,
        embedder: EmbeddingModel | None = None,
        cache_redis_url: str | None = None,
        gemini_api_key: str | None = None,
    ):
        self.storage = storage
        self.config = config or RetrievalConfig()
        self.embedder = embedder or MultilingualE5Embedder()

        # Initialize components
        self.sparse_encoder = BM25Encoder()
        self.similarity_calc = SimilarityCalculator(self.embedder, self.sparse_encoder)
        self.cache = RetrievalCache(cache_redis_url, self.config.cache_ttl)
        self.gemini_search = GeminiSearchIntegration(gemini_api_key) if gemini_api_key else None

        # Initialize retrievers
        self._init_retrievers()

        # Statistics
        self.retrieval_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_retrieval_time": 0,
            "strategy_usage": defaultdict(int),
            "namespace_usage": defaultdict(int),
        }

    def _init_retrievers(self):
        """Initialize Pinecone-based retrieval strategies"""
        self.retrievers = {
            RetrievalStrategy.QUERY_BASED: QueryBasedPineconeRetriever(
                self.storage, self.config, self.embedder, self.similarity_calc
            ),
            RetrievalStrategy.HYBRID: HybridPineconeRetriever(
                self.storage, self.config, self.embedder, self.similarity_calc
            ),
            RetrievalStrategy.NAMESPACE_AWARE: NamespaceAwareRetriever(
                self.storage, self.config, self.embedder, self.similarity_calc
            ),
            RetrievalStrategy.CASCADING: CascadingPineconeRetriever(
                self.storage, self.config, self.embedder, self.similarity_calc
            ),
        }

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
        namespace: str | None = None,
        use_external_search: bool = False,
    ) -> list[RetrievalResult]:
        """Main retrieval method"""

        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.top_k

        self.retrieval_stats["total_queries"] += 1
        self.retrieval_stats["strategy_usage"][strategy.name] += 1

        # Override namespace if specified
        original_namespaces: list[str] | None = None
        if namespace:
            original_namespaces = self.config.search_namespaces
            self.config.search_namespaces = [namespace]
            self.retrieval_stats["namespace_usage"][namespace] += 1

        # Check cache if enabled
        cache_namespace = namespace or "all"
        if self.config.use_caching:
            cached_results = await self.cache.get(query, strategy.name, top_k, cache_namespace)
            if cached_results:
                self.retrieval_stats["cache_hits"] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                if original_namespaces is not None:
                    self.config.search_namespaces = original_namespaces
                return cached_results
            self.retrieval_stats["cache_misses"] += 1

        # External search augmentation if enabled
        external_patterns = []
        if use_external_search and self.gemini_search:
            try:
                external_patterns = await self.gemini_search.search_similar_tasks(query)
                logger.info(f"Found {len(external_patterns)} external patterns")
            except Exception as e:
                logger.warning(f"External search failed: {e}")

        # Retrieve using selected strategy
        retriever = self.retrievers.get(strategy)
        if not retriever:
            logger.warning(f"Unknown strategy {strategy}, using default")
            retriever = self.retrievers[RetrievalStrategy.HYBRID]

        results = await retriever.retrieve(query, top_k)

        # Restore original namespaces
        if original_namespaces is not None:
            self.config.search_namespaces = original_namespaces

        # Augment with external patterns if available
        if external_patterns:
            results = await self._augment_with_external(results, external_patterns)

        # Cache results
        if self.config.use_caching and results:
            await self.cache.set(query, strategy.name, top_k, results, cache_namespace)

        # Update statistics
        if results:
            avg_time = np.mean([r.retrieval_time for r in results])
            self.retrieval_stats["average_retrieval_time"] = (
                self.retrieval_stats["average_retrieval_time"] * (self.retrieval_stats["total_queries"] - 1) + avg_time
            ) / self.retrieval_stats["total_queries"]

        logger.info(f"Retrieved {len(results)} memories using {strategy.name}")
        return results

    async def _augment_with_external(
        self, results: list[RetrievalResult], external_patterns: list[dict[str, Any]]
    ) -> list[RetrievalResult]:
        """Augment retrieval results with external patterns"""

        for i, result in enumerate(results):
            if i < len(external_patterns):
                result.metadata["external_pattern"] = external_patterns[i]
                result.metadata["augmented"] = True

        return results

    async def batch_retrieve(
        self,
        queries: list[str],
        strategy: RetrievalStrategy | None = None,
        top_k: int | None = None,
        parallel: bool = True,
    ) -> dict[str, list[RetrievalResult]]:
        """Batch retrieval for multiple queries"""

        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.top_k

        results = {}

        if parallel:
            # Parallel processing
            tasks = []
            for query in queries:
                task = self.retrieve(query, strategy, top_k)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)

            for query, result in zip(queries, batch_results, strict=False):
                results[query] = result
        else:
            # Sequential processing
            for query in queries:
                results[query] = await self.retrieve(query, strategy, top_k)

        return results

    def update_memory_usage(self, memory_id: str, success: bool, namespace: str = "proceduralized"):
        """Update memory usage statistics in Pinecone"""
        if memory_id in self.storage.memories:
            memory = self.storage.memories[memory_id]
            memory.increment_usage(success)

            # Update Pinecone metadata
            self.storage.update_metadata(
                memory_id,
                {
                    "usage_count": memory.usage_count,
                    "success_rate": memory.success_rate,
                },
                namespace,
            )

            logger.info(
                f"Updated usage stats for memory {memory_id}: "
                f"count={memory.usage_count}, success_rate={memory.success_rate:.2f}"
            )

    def get_retrieval_statistics(self) -> dict[str, Any]:
        """Get retrieval pipeline statistics"""
        return {
            **self.retrieval_stats,
            "cache_hit_rate": (
                self.retrieval_stats["cache_hits"]
                / max(
                    1,
                    self.retrieval_stats["cache_hits"] + self.retrieval_stats["cache_misses"],
                )
            ),
            "storage_stats": self.storage.get_statistics(),
            "retrievers_available": list(self.retrievers.keys()),
            "configured_namespaces": self.config.search_namespaces,
        }

    async def optimize_retrieval_strategy(
        self, test_queries: list[str], ground_truth: list[str] | None = None
    ) -> dict[str, Any]:
        """Optimize retrieval strategy based on test queries"""

        strategy_performance = {}

        for strategy in [
            RetrievalStrategy.QUERY_BASED,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.NAMESPACE_AWARE,
            RetrievalStrategy.CASCADING,
        ]:
            scores = []
            times = []

            for query in test_queries:
                results = await self.retrieve(query, strategy, top_k=5)

                if results:
                    # Calculate performance metrics
                    avg_score = np.mean([r.score for r in results])
                    avg_time = np.mean([r.retrieval_time for r in results])

                    scores.append(avg_score)
                    times.append(avg_time)

            if scores:
                strategy_performance[strategy.name] = {
                    "avg_score": np.mean(scores),
                    "avg_time": np.mean(times),
                    "score_std": np.std(scores),
                    "time_std": np.std(times),
                }

        # Recommend best strategy
        best_strategy = max(strategy_performance.items(), key=lambda x: x[1]["avg_score"])[0]

        return {
            "performance_by_strategy": strategy_performance,
            "recommended_strategy": best_strategy,
            "test_queries_count": len(test_queries),
        }


# ============= Example Usage =============


async def example_usage():
    """Example of how to use the Memp Retrieval pipeline with Pinecone"""

    import os

    # Initialize Pinecone storage
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "your-api-key")
    storage = PineconeMemoryStorage(api_key=pinecone_api_key)

    # Create configuration
    config = RetrievalConfig(
        strategy=RetrievalStrategy.CASCADING,
        top_k=3,
        use_reranking=True,
        use_caching=True,
        use_hybrid_search=True,
        search_namespaces=["proceduralized", "script", "trajectory"],
    )

    # Initialize retrieval pipeline
    retrieval_pipeline = MemppRetrievalPipeline(
        storage=storage, config=config, gemini_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Test query
    test_query = "How to heat something and dispose of it?"

    print("\n=== Testing Pinecone Retrieval Strategies ===")

    # Test different strategies
    strategies = [
        RetrievalStrategy.QUERY_BASED,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.NAMESPACE_AWARE,
        RetrievalStrategy.CASCADING,
    ]

    for strategy in strategies:
        print(f"\n{strategy.name}:")
        results = await retrieval_pipeline.retrieve(test_query, strategy=strategy, top_k=2)
        for r in results:
            print(f"  - {r.memory.task_pattern[:50]}... (score: {r.score:.3f}, namespace: {r.namespace})")

    # Test namespace-specific retrieval
    print("\n=== Namespace-Specific Retrieval ===")
    for namespace in ["proceduralized", "script", "trajectory"]:
        results = await retrieval_pipeline.retrieve(
            test_query, strategy=RetrievalStrategy.HYBRID, top_k=2, namespace=namespace
        )
        print(f"\n{namespace}:")
        for r in results:
            print(f"  - {r.memory.task_pattern[:50]}... (score: {r.score:.3f})")

    # Get statistics
    print("\n=== Retrieval Statistics ===")
    stats = retrieval_pipeline.get_retrieval_statistics()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
