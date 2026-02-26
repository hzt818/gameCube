"""
Short-term memory module for AI Agent Core System.
Manages current session context and conversation history.
"""
import logging
from collections import deque
from datetime import datetime
from typing import Any

from app.config import settings
from app.models.schemas import Message, MessageType

logger = logging.getLogger(__name__)


class ShortTermMemoryError(Exception):
    """Exception raised for short-term memory errors."""
    pass


class ShortTermMemory:
    """
    Short-term memory for managing current session context.
    
    Features:
    - Fixed-size message buffer
    - Automatic message compression
    - Context window management
    - Message retrieval by type/time
    """
    
    def __init__(
        self,
        max_messages: int | None = None,
        compression_threshold: int | None = None
    ) -> None:
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum messages to store
            compression_threshold: Threshold for triggering compression
        """
        self.max_messages = max_messages or settings.memory_short_term_max_messages
        self.compression_threshold = compression_threshold or settings.memory_compression_threshold
        
        self._messages: deque[Message] = deque(maxlen=self.max_messages)
        self._session_metadata: dict[str, Any] = {}
        self._created_at = datetime.utcnow()
        
        logger.info(
            f"ShortTermMemory initialized: max_messages={self.max_messages}, "
            f"compression_threshold={self.compression_threshold}"
        )
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to memory.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
        logger.debug(f"Added message: role={message.role}, content_len={len(message.content)}")
        
        if len(self._messages) >= self.compression_threshold:
            self._auto_compress()
    
    def add(
        self,
        role: MessageType | str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> Message:
        """
        Convenience method to add a message.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        if isinstance(role, str):
            role = MessageType(role)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.add_message(message)
        return message
    
    def get_messages(
        self,
        limit: int | None = None,
        role_filter: MessageType | None = None
    ) -> list[Message]:
        """
        Get messages from memory.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Filter by message role
            
        Returns:
            List of messages
        """
        messages = list(self._messages)
        
        if role_filter:
            messages = [m for m in messages if m.role == role_filter]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_context(self, max_tokens: int = 4000) -> list[dict[str, str]]:
        """
        Get conversation context formatted for LLM.
        
        Args:
            max_tokens: Approximate max tokens to include
            
        Returns:
            List of message dictionaries
        """
        messages = list(self._messages)
        result: list[dict[str, str]] = []
        total_length = 0
        
        for message in reversed(messages):
            msg_length = len(message.content.split())
            
            if total_length + msg_length > max_tokens:
                break
            
            result.insert(0, {
                "role": message.role,
                "content": message.content
            })
            total_length += msg_length
        
        return result
    
    def get_last_n_messages(self, n: int) -> list[Message]:
        """
        Get last N messages.
        
        Args:
            n: Number of messages to return
            
        Returns:
            List of last N messages
        """
        return list(self._messages)[-n:]
    
    def get_messages_since(self, timestamp: datetime) -> list[Message]:
        """
        Get messages since a specific timestamp.
        
        Args:
            timestamp: Starting timestamp
            
        Returns:
            List of messages after timestamp
        """
        return [m for m in self._messages if m.timestamp >= timestamp]
    
    def search(self, query: str, case_sensitive: bool = False) -> list[Message]:
        """
        Search messages by content.
        
        Args:
            query: Search query
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching messages
        """
        if not case_sensitive:
            query = query.lower()
        
        results = []
        for message in self._messages:
            content = message.content if case_sensitive else message.content.lower()
            if query in content:
                results.append(message)
        
        return results
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set session metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._session_metadata[key] = value
        logger.debug(f"Set metadata: {key}={value}")
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get session metadata.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self._session_metadata.get(key, default)
    
    def _auto_compress(self) -> None:
        """
        Automatically compress old messages.
        Keeps system messages and recent messages.
        """
        if len(self._messages) < self.compression_threshold:
            return
        
        system_messages = [m for m in self._messages if m.role == MessageType.SYSTEM]
        recent_count = self.compression_threshold // 2
        recent_messages = list(self._messages)[-recent_count:]
        
        compressed_count = len(self._messages) - len(system_messages) - len(recent_messages)
        
        self._messages.clear()
        for msg in system_messages:
            self._messages.append(msg)
        for msg in recent_messages:
            if msg not in self._messages:
                self._messages.append(msg)
        
        logger.info(f"Auto-compressed {compressed_count} messages")
    
    def compress(self, keep_last: int = 10) -> dict[str, Any]:
        """
        Manually compress memory, keeping recent messages.
        
        Args:
            keep_last: Number of recent messages to keep
            
        Returns:
            Compression statistics
        """
        original_count = len(self._messages)
        
        system_messages = [m for m in self._messages if m.role == MessageType.SYSTEM]
        recent_messages = list(self._messages)[-keep_last:]
        
        self._messages.clear()
        for msg in system_messages:
            self._messages.append(msg)
        for msg in recent_messages:
            if msg not in self._messages:
                self._messages.append(msg)
        
        compressed_count = original_count - len(self._messages)
        
        logger.info(f"Manual compression: {original_count} -> {len(self._messages)} messages")
        
        return {
            "original_count": original_count,
            "compressed_count": compressed_count,
            "remaining_count": len(self._messages)
        }
    
    def clear(self) -> None:
        """Clear all messages and metadata."""
        self._messages.clear()
        self._session_metadata.clear()
        self._created_at = datetime.utcnow()
        logger.info("Short-term memory cleared")
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get memory summary statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        role_counts: dict[str, int] = {}
        for msg in self._messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
        
        return {
            "total_messages": len(self._messages),
            "max_messages": self.max_messages,
            "role_distribution": role_counts,
            "created_at": self._created_at.isoformat(),
            "metadata_keys": list(self._session_metadata.keys())
        }
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __repr__(self) -> str:
        return f"ShortTermMemory(messages={len(self._messages)}/{self.max_messages})"
