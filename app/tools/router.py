"""
Tool router module for AI Agent Core System.
Handles tool execution routing and concurrency.
"""
import asyncio
import logging
from typing import Any

from app.config import settings
from app.models.schemas import ToolCall, ToolResult, ToolCallStatus
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


class RouterError(Exception):
    """Exception raised for router errors."""
    pass


class ToolRouter:
    """
    Routes tool calls to appropriate handlers.
    
    Features:
    - Tool lookup and dispatch
    - Concurrent execution
    - Timeout management
    - Error handling
    """
    
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        default_timeout: float | None = None,
        max_concurrent: int = 10
    ) -> None:
        """
        Initialize tool router.
        
        Args:
            registry: Tool registry to use
            default_timeout: Default timeout for tool execution
            max_concurrent: Maximum concurrent tool executions
        """
        self.registry = registry or get_registry()
        self.default_timeout = default_timeout or settings.agent_tool_timeout
        self.max_concurrent = max_concurrent
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._execution_count = 0
        self._error_count = 0
        
        logger.info(
            f"ToolRouter initialized: max_concurrent={max_concurrent}, "
            f"default_timeout={self.default_timeout}"
        )
    
    def _get_tool(self, tool_name: str) -> BaseTool:
        """
        Get tool from registry.
        
        Args:
            tool_name: Name of tool
            
        Returns:
            BaseTool instance
            
        Raises:
            RouterError: If tool not found
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            raise RouterError(f"Tool not found: {tool_name}")
        return tool
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call request
            
        Returns:
            ToolResult: Execution result
        """
        async with self._semaphore:
            self._execution_count += 1
            
            try:
                tool = self._get_tool(tool_call.tool_name)
                result = await tool.execute(tool_call)
                
                if result.status == ToolCallStatus.FAILED:
                    self._error_count += 1
                
                return result
                
            except RouterError as e:
                self._error_count += 1
                logger.error(f"Router error: {e}")
                
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    status=ToolCallStatus.FAILED,
                    error=str(e)
                )
            
            except Exception as e:
                self._error_count += 1
                logger.error(f"Unexpected error executing tool: {e}")
                
                return ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    status=ToolCallStatus.FAILED,
                    error=f"Unexpected error: {e}"
                )
    
    async def execute_batch(
        self,
        tool_calls: list[ToolCall],
        fail_fast: bool = False
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls concurrently.
        
        Args:
            tool_calls: List of tool calls
            fail_fast: If True, stop on first failure
            
        Returns:
            List of tool results in same order as calls
        """
        if not tool_calls:
            return []
        
        if fail_fast:
            results = []
            for call in tool_calls:
                result = await self.execute(call)
                results.append(result)
                if result.status != ToolCallStatus.SUCCESS:
                    break
            return results
        
        tasks = [self.execute(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult(
                    call_id=tool_calls[i].call_id,
                    tool_name=tool_calls[i].tool_name,
                    status=ToolCallStatus.FAILED,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_with_retry(
        self,
        tool_call: ToolCall,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> ToolResult:
        """
        Execute with automatic retry on failure.
        
        Args:
            tool_call: Tool call request
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            
        Returns:
            ToolResult: Final execution result
        """
        last_result: ToolResult | None = None
        
        for attempt in range(max_retries):
            result = await self.execute(tool_call)
            
            if result.status == ToolCallStatus.SUCCESS:
                return result
            
            last_result = result
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"Tool {tool_call.tool_name} failed (attempt {attempt + 1}), "
                    f"retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
        
        return last_result or ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            status=ToolCallStatus.FAILED,
            error="All retry attempts failed"
        )
    
    def get_tools_description(self) -> str:
        """
        Get formatted description of all available tools.
        
        Returns:
            Formatted string describing all tools
        """
        return self.registry.get_tools_description()
    
    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """
        Get JSON schema for a specific tool.
        
        Args:
            tool_name: Name of tool
            
        Returns:
            Tool schema or None
        """
        tool = self.registry.get(tool_name)
        return tool.get_schema() if tool else None
    
    def get_all_schemas(self) -> list[dict[str, Any]]:
        """
        Get JSON schemas for all tools.
        
        Returns:
            List of all tool schemas
        """
        return self.registry.get_schemas()
    
    def list_tools(self) -> list[str]:
        """
        List all available tools.
        
        Returns:
            List of tool names
        """
        return self.registry.list_tools()
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if tool is available.
        
        Args:
            tool_name: Name of tool
            
        Returns:
            True if tool exists
        """
        return self.registry.has_tool(tool_name)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._execution_count - self._error_count) / self._execution_count
                if self._execution_count > 0 else 0
            ),
            "max_concurrent": self.max_concurrent,
            "registry_stats": self.registry.get_stats()
        }
    
    def __repr__(self) -> str:
        return f"ToolRouter(tools={len(self.registry)}, executions={self._execution_count})"


_default_router: ToolRouter | None = None


def get_router() -> ToolRouter:
    """
    Get the default tool router.
    
    Returns:
        Default ToolRouter instance
    """
    global _default_router
    if _default_router is None:
        _default_router = ToolRouter()
    return _default_router
