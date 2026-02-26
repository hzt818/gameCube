"""
RAG (Retrieval-Augmented Generation) module for AI Agent Core System.
Provides knowledge retrieval capabilities.
"""
import logging
from typing import Any

from app.config import settings
from app.memory.vector_memory import VectorMemory
from app.models.schemas import MemoryEntry, RAGQuery, RAGResponse, RAGResult

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for RAG errors."""
    pass


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        raise NotImplementedError
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
    
    async def embed(self, text: str) -> list[float]:
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(val)
        
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


class RAG:
    """
    Retrieval-Augmented Generation module.
    
    Features:
    - Document indexing with embeddings
    - Semantic search
    - Context building for LLM
    - Knowledge base management
    """
    
    def __init__(
        self,
        vector_memory: VectorMemory | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None
    ) -> None:
        """
        Initialize RAG module.
        
        Args:
            vector_memory: Vector memory for storage
            embedding_provider: Provider for generating embeddings
            top_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity threshold
        """
        self.vector_memory = vector_memory or VectorMemory()
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.top_k = top_k or settings.rag_top_k
        self.similarity_threshold = similarity_threshold or settings.rag_similarity_threshold
        
        self._initialized = False
        
        logger.info(
            f"RAG initialized: top_k={self.top_k}, "
            f"similarity_threshold={self.similarity_threshold}"
        )
    
    async def initialize(self) -> None:
        """Initialize RAG components."""
        if self._initialized:
            return
        
        await self.vector_memory.initialize()
        self._initialized = True
        logger.info("RAG initialized")
    
    async def close(self) -> None:
        """Close RAG components."""
        await self.vector_memory.close()
        self._initialized = False
        logger.info("RAG closed")
    
    async def index_document(
        self,
        content: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5
    ) -> str:
        """
        Index a document for retrieval.
        
        Args:
            content: Document content
            session_id: Session identifier
            metadata: Optional metadata
            importance: Document importance score
            
        Returns:
            Document entry ID
        """
        if not self._initialized:
            await self.initialize()
        
        embedding = await self.embedding_provider.embed(content)
        
        entry = MemoryEntry(
            session_id=session_id,
            content=content,
            metadata=metadata or {},
            importance=importance
        )
        
        entry_id = await self.vector_memory.store(entry, embedding)
        
        logger.debug(f"Indexed document: {entry_id}")
        return entry_id
    
    async def index_documents(
        self,
        documents: list[dict[str, Any]],
        session_id: str
    ) -> list[str]:
        """
        Index multiple documents.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            session_id: Session identifier
            
        Returns:
            List of entry IDs
        """
        if not self._initialized:
            await self.initialize()
        
        entries_with_embeddings = []
        
        contents = [doc.get("content", "") for doc in documents]
        embeddings = await self.embedding_provider.embed_batch(contents)
        
        entry_ids = []
        for doc, embedding in zip(documents, embeddings):
            entry = MemoryEntry(
                session_id=session_id,
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                importance=doc.get("importance", 0.5)
            )
            entries_with_embeddings.append((entry, embedding))
        
        if entries_with_embeddings:
            entry_ids = await self.vector_memory.store_batch(entries_with_embeddings)
        
        logger.info(f"Indexed {len(entry_ids)} documents")
        return entry_ids
    
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: dict[str, Any] | None = None
    ) -> RAGResponse:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (uses default if None)
            similarity_threshold: Minimum similarity (uses default if None)
            filters: Optional metadata filters
            
        Returns:
            RAGResponse with results
        """
        if not self._initialized:
            await self.initialize()
        
        top_k = top_k or self.top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        query_embedding = await self.embedding_provider.embed(query)
        
        results = await self.vector_memory.search(
            query_vector=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters
        )
        
        rag_results = [
            RAGResult(
                content=entry.content,
                score=score,
                metadata=entry.metadata
            )
            for entry, score in results
        ]
        
        logger.debug(f"Retrieved {len(rag_results)} results for query")
        
        return RAGResponse(
            query=query,
            results=rag_results,
            total_results=len(rag_results)
        )
    
    async def retrieve_with_query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Retrieve using RAGQuery model.
        
        Args:
            rag_query: RAG query model
            
        Returns:
            RAGResponse with results
        """
        return await self.retrieve(
            query=rag_query.query,
            top_k=rag_query.top_k,
            similarity_threshold=rag_query.similarity_threshold,
            filters=rag_query.filters
        )
    
    async def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        top_k: int | None = None
    ) -> str:
        """
        Build context string for LLM from retrieved documents.
        
        Args:
            query: Search query
            max_tokens: Approximate max tokens for context
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        response = await self.retrieve(query, top_k=top_k)
        
        if not response.results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(response.results):
            doc_text = f"[Document {i + 1}] (relevance: {result.score:.2f})\n{result.content}\n"
            
            doc_length = len(doc_text.split())
            if total_length + doc_length > max_tokens:
                break
            
            context_parts.append(doc_text)
            total_length += doc_length
        
        context = "\n".join(context_parts)
        return f"Retrieved Context:\n{context}"
    
    async def hybrid_search(
        self,
        query: str,
        keyword: str | None = None,
        top_k: int | None = None
    ) -> RAGResponse:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Semantic search query
            keyword: Optional keyword filter
            top_k: Number of results
            
        Returns:
            RAGResponse with results
        """
        filters = None
        if keyword:
            filters = {"keyword": keyword}
        
        return await self.retrieve(query, top_k=top_k, filters=filters)
    
    async def delete_document(self, entry_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            entry_id: Document entry ID
            
        Returns:
            True if deleted
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.vector_memory.delete(entry_id)
    
    async def clear_session(self, session_id: str) -> int:
        """
        Clear all documents for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of documents cleared
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.vector_memory.delete_by_session(session_id)
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get RAG statistics.
        
        Returns:
            Dictionary with stats
        """
        if not self._initialized:
            return {"initialized": False}
        
        collection_info = await self.vector_memory.get_collection_info()
        
        return {
            "initialized": self._initialized,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "collection_info": collection_info
        }
    
    def __repr__(self) -> str:
        return f"RAG(initialized={self._initialized}, top_k={self.top_k})"
