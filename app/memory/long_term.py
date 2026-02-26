"""
Long-term memory module for AI Agent Core System.
Manages persistent storage using PostgreSQL.
"""
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    delete,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.models.schemas import MemoryEntry

logger = logging.getLogger(__name__)


class LongTermMemoryError(Exception):
    """Exception raised for long-term memory errors."""
    pass


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class MemoryRecord(Base):
    """Memory record model for PostgreSQL."""
    
    __tablename__ = "memory_records"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entry_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), default="text")
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    
    __table_args__ = (
        Index("ix_memory_records_session_created", "session_id", "created_at"),
        Index("ix_memory_records_user_created", "user_id", "created_at"),
    )


class SessionRecord(Base):
    """Session record model for PostgreSQL."""
    
    __tablename__ = "session_records"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LongTermMemory:
    """
    Long-term memory for persistent storage.
    
    Features:
    - PostgreSQL-based persistence
    - Session management
    - Memory archival
    - Expiration handling
    """
    
    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None
    ) -> None:
        """
        Initialize long-term memory.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Max overflow connections
        """
        self.database_url = database_url or settings.database_url
        self.pool_size = pool_size or settings.database_pool_size
        self.max_overflow = max_overflow or settings.database_max_overflow
        
        self._engine = None
        self._session_factory: async_sessionmaker | None = None
        self._initialized = False
        
        logger.info(f"LongTermMemory configured: pool_size={self.pool_size}")
    
    async def initialize(self) -> None:
        """Initialize database connection and tables."""
        if self._initialized:
            return
        
        try:
            self._engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=settings.debug
            )
            
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
            logger.info("LongTermMemory initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise LongTermMemoryError(f"Database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("LongTermMemory connection closed")
    
    def _get_session(self) -> AsyncSession:
        """Get database session."""
        if not self._session_factory:
            raise LongTermMemoryError("Database not initialized")
        return self._session_factory()
    
    async def store(self, entry: MemoryEntry, user_id: str | None = None) -> str:
        """
        Store a memory entry.
        
        Args:
            entry: Memory entry to store
            user_id: Optional user ID
            
        Returns:
            Entry ID
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                record = MemoryRecord(
                    entry_id=entry.entry_id,
                    session_id=entry.session_id,
                    user_id=user_id,
                    content=entry.content,
                    importance=entry.importance,
                    metadata=entry.metadata,
                    expires_at=entry.expires_at
                )
                
                session.add(record)
                await session.commit()
                
                logger.debug(f"Stored memory entry: {entry.entry_id}")
                return entry.entry_id
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store entry: {e}")
                raise LongTermMemoryError(f"Storage failed: {e}")
    
    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """
        Get entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            MemoryEntry or None
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = select(MemoryRecord).where(
                    MemoryRecord.entry_id == entry_id,
                    MemoryRecord.is_archived == False
                )
                
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                
                if not record:
                    return None
                
                return MemoryEntry(
                    entry_id=record.entry_id,
                    session_id=record.session_id,
                    content=record.content,
                    metadata=record.metadata,
                    importance=record.importance,
                    created_at=record.created_at,
                    expires_at=record.expires_at
                )
                
            except Exception as e:
                logger.error(f"Failed to get entry: {e}")
                return None
    
    async def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[MemoryEntry]:
        """
        Get entries by session ID.
        
        Args:
            session_id: Session ID
            limit: Maximum entries to return
            offset: Offset for pagination
            
        Returns:
            List of memory entries
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = (
                    select(MemoryRecord)
                    .where(
                        MemoryRecord.session_id == session_id,
                        MemoryRecord.is_archived == False
                    )
                    .order_by(MemoryRecord.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                return [
                    MemoryEntry(
                        entry_id=r.entry_id,
                        session_id=r.session_id,
                        content=r.content,
                        metadata=r.metadata,
                        importance=r.importance,
                        created_at=r.created_at,
                        expires_at=r.expires_at
                    )
                    for r in records
                ]
                
            except Exception as e:
                logger.error(f"Failed to get session entries: {e}")
                return []
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[MemoryEntry]:
        """
        Get entries by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum entries to return
            offset: Offset for pagination
            
        Returns:
            List of memory entries
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = (
                    select(MemoryRecord)
                    .where(
                        MemoryRecord.user_id == user_id,
                        MemoryRecord.is_archived == False
                    )
                    .order_by(MemoryRecord.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                return [
                    MemoryEntry(
                        entry_id=r.entry_id,
                        session_id=r.session_id,
                        content=r.content,
                        metadata=r.metadata,
                        importance=r.importance,
                        created_at=r.created_at,
                        expires_at=r.expires_at
                    )
                    for r in records
                ]
                
            except Exception as e:
                logger.error(f"Failed to get user entries: {e}")
                return []
    
    async def search(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int = 50
    ) -> list[MemoryEntry]:
        """
        Search entries by content.
        
        Args:
            query: Search query
            session_id: Optional session filter
            user_id: Optional user filter
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = select(MemoryRecord).where(
                    MemoryRecord.content.ilike(f"%{query}%"),
                    MemoryRecord.is_archived == False
                )
                
                if session_id:
                    stmt = stmt.where(MemoryRecord.session_id == session_id)
                if user_id:
                    stmt = stmt.where(MemoryRecord.user_id == user_id)
                
                stmt = stmt.order_by(MemoryRecord.importance.desc()).limit(limit)
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                return [
                    MemoryEntry(
                        entry_id=r.entry_id,
                        session_id=r.session_id,
                        content=r.content,
                        metadata=r.metadata,
                        importance=r.importance,
                        created_at=r.created_at,
                        expires_at=r.expires_at
                    )
                    for r in records
                ]
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []
    
    async def update_importance(self, entry_id: str, importance: float) -> bool:
        """
        Update entry importance.
        
        Args:
            entry_id: Entry ID
            importance: New importance value
            
        Returns:
            True if updated
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = (
                    update(MemoryRecord)
                    .where(MemoryRecord.entry_id == entry_id)
                    .values(importance=importance, updated_at=datetime.utcnow())
                )
                
                await session.execute(stmt)
                await session.commit()
                
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update importance: {e}")
                return False
    
    async def archive(self, entry_id: str) -> bool:
        """
        Archive an entry.
        
        Args:
            entry_id: Entry ID to archive
            
        Returns:
            True if archived
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = (
                    update(MemoryRecord)
                    .where(MemoryRecord.entry_id == entry_id)
                    .values(is_archived=True, updated_at=datetime.utcnow())
                )
                
                await session.execute(stmt)
                await session.commit()
                
                logger.debug(f"Archived entry: {entry_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to archive entry: {e}")
                return False
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete an entry.
        
        Args:
            entry_id: Entry ID to delete
            
        Returns:
            True if deleted
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = delete(MemoryRecord).where(MemoryRecord.entry_id == entry_id)
                
                await session.execute(stmt)
                await session.commit()
                
                logger.debug(f"Deleted entry: {entry_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete entry: {e}")
                return False
    
    async def delete_session(self, session_id: str) -> int:
        """
        Delete all entries for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of deleted entries
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                stmt = delete(MemoryRecord).where(
                    MemoryRecord.session_id == session_id
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Deleted {deleted_count} entries for session: {session_id}")
                return deleted_count
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to delete session entries: {e}")
                return 0
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of removed entries
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                now = datetime.utcnow()
                
                stmt = delete(MemoryRecord).where(
                    MemoryRecord.expires_at != None,
                    MemoryRecord.expires_at < now
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} expired entries")
                return deleted_count
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to cleanup expired entries: {e}")
                return 0
    
    async def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with stats
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._get_session() as session:
            try:
                from sqlalchemy import func
                
                total_stmt = select(func.count()).select_from(MemoryRecord)
                total_result = await session.execute(total_stmt)
                total_count = total_result.scalar()
                
                active_stmt = select(func.count()).select_from(MemoryRecord).where(
                    MemoryRecord.is_archived == False
                )
                active_result = await session.execute(active_stmt)
                active_count = active_result.scalar()
                
                return {
                    "total_entries": total_count,
                    "active_entries": active_count,
                    "archived_entries": total_count - active_count
                }
                
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {"error": str(e)}
    
    def __repr__(self) -> str:
        return f"LongTermMemory(initialized={self._initialized})"
