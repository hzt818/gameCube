"""
Data models and schemas for AI Agent Core System.
Defines Pydantic models for requests, responses, and internal data structures.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentState(str, Enum):
    """Agent execution states for state machine."""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class MessageType(str, Enum):
    """Message types for conversation history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ERROR = "error"


class ToolCallStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Message(BaseModel):
    """Single message in conversation history."""
    role: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class ToolCall(BaseModel):
    """Represents a tool call request."""
    tool_name: str = Field(..., min_length=1, description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique call identifier")


class ToolResult(BaseModel):
    """Result of a tool execution."""
    call_id: str = Field(..., description="ID of the corresponding tool call")
    tool_name: str = Field(..., description="Name of the executed tool")
    status: ToolCallStatus = Field(..., description="Execution status")
    result: Any = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


class ReasoningOutput(BaseModel):
    """Output from the reasoning core."""
    thought: str = Field(..., description="Agent's reasoning/thought process")
    action: str = Field(..., description="Tool name or 'final_answer'")
    action_input: dict[str, Any] = Field(default_factory=dict, description="Input for the action")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    requires_reflection: bool = Field(default=False, description="Whether reflection is needed")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is not empty."""
        if not v or not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class TaskPlan(BaseModel):
    """Represents a planned task."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(..., min_length=1, description="Task description")
    dependencies: list[str] = Field(default_factory=list, description="List of dependent task IDs")
    priority: int = Field(default=0, ge=0, le=10, description="Task priority (0-10)")
    status: AgentState = Field(default=AgentState.IDLE, description="Current task status")
    estimated_steps: int = Field(default=1, ge=1, description="Estimated steps to complete")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


class SessionContext(BaseModel):
    """Context for an agent session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = Field(default=None, description="User identifier")
    conversation_history: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    current_state: AgentState = Field(default=AgentState.IDLE)
    iteration_count: int = Field(default=0, ge=0)
    total_tool_calls: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class AgentRequest(BaseModel):
    """Request model for agent execution."""
    query: str = Field(..., min_length=1, description="User query or task")
    session_id: str | None = Field(default=None, description="Optional session ID for continuity")
    user_id: str | None = Field(default=None, description="Optional user identifier")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    max_iterations: int | None = Field(default=None, ge=1, description="Override max iterations")
    tools: list[str] | None = Field(default=None, description="Specific tools to use")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Agent's final response")
    thought_process: list[str] = Field(default_factory=list, description="List of thoughts")
    tool_calls_made: int = Field(default=0, description="Number of tool calls made")
    iterations_used: int = Field(default=0, description="Number of iterations used")
    status: AgentState = Field(..., description="Final agent state")
    success: bool = Field(..., description="Whether execution was successful")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


class MemoryEntry(BaseModel):
    """Entry in agent memory."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., description="Associated session ID")
    content: str = Field(..., min_length=1, description="Memory content")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory importance score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = Field(default=None, description="Optional expiration time")


class RAGQuery(BaseModel):
    """Query model for RAG retrieval."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity")
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters")


class RAGResult(BaseModel):
    """Single RAG retrieval result."""
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Response model for RAG retrieval."""
    query: str = Field(..., description="Original query")
    results: list[RAGResult] = Field(default_factory=list, description="Retrieved results")
    total_results: int = Field(default=0, description="Total number of results")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict, description="Component statuses")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
