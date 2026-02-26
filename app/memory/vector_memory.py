"""
Vector memory module for AI Agent Core System.
Manages vector embeddings and similarity search using Qdrant.
"""
import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.models.schemas import MemoryEntry

logger = logging.getLogger(__name__)


class VectorMemoryError(Exception):
    """Exception raised for vector memory errors."""
    pass


class VectorMemory:
    """
    Vector memory for semantic search and retrieval.
    
    Features:
    - Qdrant-based vector storage
    - Similarity search
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
        vector_dimension: int | None = None
    ) -> None:
        """
        Initialize vector memory.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Collection name
            api_key: Qdrant API key
            vector_dimension: Vector embedding dimension
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.api_key = api_key or settings.qdrant_api_key
        self.vector_dimension = vector_dimension or settings.memory_vector_dimension
        
        self._client: AsyncQdrantClient | None = None
        self._initialized = False
        
        logger.info(
            f"VectorMemory configured: host={self.host}, port={self.port}, "
            f"collection={self.collection_name}, dimension={self.vector_dimension}"
        )
    
    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )
        return self._client
    
    async def initialize(self) -> None:
        """Initialize collection if not exists."""
        client = await self._get_client()
        
        try:
            collections = await client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise VectorMemoryError(f"Initialization failed: {e}")
    
    async def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("VectorMemory client closed")
    
    async def store(
        self,
        entry: MemoryEntry,
        embedding: list[float] | None = None
    ) -> str:
        """
        Store a memory entry with embedding.
        
        Args:
            entry: Memory entry to store
            embedding: Vector embedding (if None, entry.embedding is used)
            
        Returns:
            Entry ID
            
        Raises:
            VectorMemoryError: If storage fails
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        vector = embedding or entry.embedding
        if vector is None:
            raise VectorMemoryError("No embedding provided for storage")
        
        if len(vector) != self.vector_dimension:
            raise VectorMemoryError(
                f"Vector dimension mismatch: expected {self.vector_dimension}, "
                f"got {len(vector)}"
            )
        
        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=entry.entry_id,
                        vector=vector,
                        payload={
                            "session_id": entry.session_id,
                            "content": entry.content,
                            "importance": entry.importance,
                            "created_at": entry.created_at.isoformat(),
                            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                            **entry.metadata
                        }
                    )
                ]
            )
            
            logger.debug(f"Stored entry: {entry.entry_id}")
            return entry.entry_id
            
        except Exception as e:
            logger.error(f"Failed to store entry: {e}")
            raise VectorMemoryError(f"Storage failed: {e}")
    
    async def store_batch(
        self,
        entries: list[tuple[MemoryEntry, list[float]]]
    ) -> list[str]:
        """
        Store multiple entries in batch.
        
        Args:
            entries: List of (entry, embedding) tuples
            
        Returns:
            List of stored entry IDs
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        points = []
        entry_ids = []
        
        for entry, embedding in entries:
            if len(embedding) != self.vector_dimension:
                logger.warning(f"Skipping entry with wrong dimension: {len(embedding)}")
                continue
            
            points.append(
                models.PointStruct(
                    id=entry.entry_id,
                    vector=embedding,
                    payload={
                        "session_id": entry.session_id,
                        "content": entry.content,
                        "importance": entry.importance,
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        **entry.metadata
                    }
                )
            )
            entry_ids.append(entry.entry_id)
        
        if not points:
            return []
        
        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} entries in batch")
            return entry_ids
            
        except Exception as e:
            logger.error(f"Batch storage failed: {e}")
            raise VectorMemoryError(f"Batch storage failed: {e}")
    
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filters: dict[str, Any] | None = None
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search for similar entries.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filters: Metadata filters
            
        Returns:
            List of (entry, score) tuples
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        filter_conditions = None
        if filters:
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )
        
        try:
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=similarity_threshold,
                query_filter=filter_conditions
            )
            
            entries = []
            for result in results:
                payload = result.payload or {}
                
                entry = MemoryEntry(
                    entry_id=str(result.id),
                    session_id=payload.get("session_id", ""),
                    content=payload.get("content", ""),
                    embedding=query_vector,
                    metadata={k: v for k, v in payload.items()
                             if k not in ["session_id", "content", "importance", "created_at", "expires_at"]},
                    importance=payload.get("importance", 0.5),
                    created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.utcnow(),
                    expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None
                )
                entries.append((entry, result.score))
            
            logger.debug(f"Search returned {len(entries)} results")
            return entries
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorMemoryError(f"Search failed: {e}")
    
    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """
        Get entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            MemoryEntry or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        try:
            result = await client.retrieve(
                collection_name=self.collection_name,
                ids=[entry_id]
            )
            
            if not result:
                return None
            
            payload = result[0].payload or {}
            
            return MemoryEntry(
                entry_id=str(result[0].id),
                session_id=payload.get("session_id", ""),
                content=payload.get("content", ""),
                metadata={k: v for k, v in payload.items()
                         if k not in ["session_id", "content", "importance", "created_at", "expires_at"]},
                importance=payload.get("importance", 0.5),
                created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.utcnow(),
                expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None
            )
            
        except UnexpectedResponse:
            return None
        except Exception as e:
            logger.error(f"Failed to get entry: {e}")
            return None
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete entry by ID.
        
        Args:
            entry_id: Entry ID to delete
            
        Returns:
            True if deleted successfully
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[entry_id]
                )
            )
            
            logger.debug(f"Deleted entry: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entry: {e}")
            return False
    
    async def delete_by_session(self, session_id: str) -> int:
        """
        Delete all entries for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of deleted entries
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self._get_client()
        
        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=session_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted entries for session: {session_id}")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete session entries: {e}")
            return 0
    
    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get collection information.
        
        Returns:
            Dictionary with collection stats
        """
        client = await self._get_client()
        
        try:
            info = await client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "vector_dimension": self.vector_dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def __repr__(self) -> str:
        return f"VectorMemory(collection={self.collection_name}, dimension={self.vector_dimension})"


from datetime import datetime
