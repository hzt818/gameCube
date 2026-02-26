"""
Base tool module for AI Agent Core System.
Defines the abstract base class for all tools.
"""
import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from app.models.schemas import ToolCall, ToolResult, ToolCallStatus

logger = logging.getLogger(__name__)


class ToolValidationError(Exception):
    """Exception raised for tool validation errors."""
    pass


class ToolExecutionError(Exception):
    """Exception raised for tool execution errors."""
    pass


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., min_length=1, description="Parameter name")
    type: str = Field(default="string", description="Parameter type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not required")
    enum: list[Any] | None = Field(default=None, description="Allowed values for enum type")


class ToolMetadata(BaseModel):
    """Metadata for a tool."""
    name: str = Field(..., min_length=1, description="Tool name")
    description: str = Field(default="", description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    author: str = Field(default="", description="Tool author")
    tags: list[str] = Field(default_factory=list, description="Tool tags")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    returns: str = Field(default="string", description="Return type description")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    timeout: float = Field(default=30.0, description="Default timeout in seconds")
    dangerous: bool = Field(default=False, description="Whether tool has side effects")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores or hyphens")
        return v.lower()


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    All tools must inherit from this class and implement:
    - _execute: The actual tool logic
    - metadata: Tool metadata definition
    
    Features:
    - Parameter validation
    - Timeout handling
    - Error handling
    - Logging
    """
    
    def __init__(self, timeout: float | None = None) -> None:
        """
        Initialize tool.
        
        Args:
            timeout: Optional timeout override
        """
        self._timeout = timeout
        self._execution_count = 0
        self._error_count = 0
        logger.info(f"Initialized tool: {self.metadata.name}")
    
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """
        Get tool metadata.
        
        Returns:
            ToolMetadata: Tool metadata definition
        """
        pass
    
    @abstractmethod
    async def _execute(self, **kwargs: Any) -> Any:
        """
        Execute tool logic. Must be implemented by subclasses.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def get_timeout(self) -> float:
        """Get tool timeout."""
        return self._timeout or self.metadata.timeout
    
    def validate_parameters(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and sanitize input parameters.
        
        Args:
            arguments: Input arguments
            
        Returns:
            Validated arguments
            
        Raises:
            ToolValidationError: If validation fails
        """
        validated: dict[str, Any] = {}
        params_by_name = {p.name: p for p in self.metadata.parameters}
        
        for param in self.metadata.parameters:
            if param.name in arguments:
                validated[param.name] = arguments[param.name]
            elif param.required:
                raise ToolValidationError(
                    f"Missing required parameter: {param.name}"
                )
            elif param.default is not None:
                validated[param.name] = param.default
        
        for key in arguments:
            if key not in params_by_name:
                logger.warning(f"Unknown parameter '{key}' for tool {self.metadata.name}")
        
        return validated
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute tool with full error handling and logging.
        
        Args:
            tool_call: Tool call request
            
        Returns:
            ToolResult: Execution result
        """
        start_time = datetime.utcnow()
        self._execution_count += 1
        
        logger.info(
            f"Executing tool: {self.metadata.name} "
            f"(call_id={tool_call.call_id})"
        )
        
        try:
            validated_args = self.validate_parameters(tool_call.arguments)
            
            result = await asyncio.wait_for(
                self._execute(**validated_args),
                timeout=self.get_timeout()
            )
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Tool {self.metadata.name} completed successfully "
                f"(time={execution_time:.2f}s)"
            )
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=self.metadata.name,
                status=ToolCallStatus.SUCCESS,
                result=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            self._error_count += 1
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Tool {self.metadata.name} timed out after {self.get_timeout()}s"
            )
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=self.metadata.name,
                status=ToolCallStatus.TIMEOUT,
                error=f"Execution timed out after {self.get_timeout()} seconds",
                execution_time=execution_time
            )
            
        except ToolValidationError as e:
            self._error_count += 1
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Tool {self.metadata.name} validation error: {e}")
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=self.metadata.name,
                status=ToolCallStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
            
        except Exception as e:
            self._error_count += 1
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Tool {self.metadata.name} execution error: {e}")
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=self.metadata.name,
                status=ToolCallStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_schema(self) -> dict[str, Any]:
        """
        Get JSON schema for the tool.
        
        Returns:
            Dictionary with tool schema
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        
        for param in self.metadata.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "returns": self.metadata.returns,
            "dangerous": self.metadata.dangerous
        }
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get tool execution statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "name": self.metadata.name,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._execution_count - self._error_count) / self._execution_count
                if self._execution_count > 0 else 0
            )
        }
    
    def __repr__(self) -> str:
        return f"Tool(name={self.metadata.name}, version={self.metadata.version})"


class FunctionTool(BaseTool):
    """
    Tool wrapper for simple functions.
    
    Allows wrapping regular Python functions as tools.
    """
    
    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter] | None = None,
        timeout: float | None = None
    ) -> None:
        """
        Initialize function tool.
        
        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description
            parameters: Parameter definitions
            timeout: Execution timeout
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""
        self._parameters = parameters or self._extract_parameters(func)
        self._timeout = timeout
        
        super().__init__(timeout=timeout)
    
    def _extract_parameters(self, func: Callable) -> list[ToolParameter]:
        """Extract parameters from function signature."""
        params = []
        sig = inspect.signature(func)
        
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                param_type = type_map.get(param.annotation, "string")
            
            params.append(ToolParameter(
                name=name,
                type=param_type,
                required=param.default == inspect.Parameter.empty,
                default=param.default if param.default != inspect.Parameter.empty else None
            ))
        
        return params
    
    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            timeout=self._timeout or 30.0
        )
    
    async def _execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        else:
            return self._func(**kwargs)


def tool(
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator to create a FunctionTool from a function.
    
    Args:
        name: Tool name
        description: Tool description
        timeout: Execution timeout
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            timeout=timeout
        )
    return decorator
