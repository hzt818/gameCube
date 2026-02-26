"""
Tool registry module for AI Agent Core System.
Manages tool registration and discovery.
"""
import logging
from typing import Any, Callable

from app.tools.base import BaseTool, FunctionTool, ToolMetadata

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry errors."""
    pass


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Features:
    - Tool registration and discovery
    - Category management
    - Tool validation
    - Bulk operations
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, set[str]] = {}
        self._aliases: dict[str, str] = {}
        
        logger.info("ToolRegistry initialized")
    
    def register(
        self,
        tool: BaseTool,
        categories: list[str] | None = None,
        aliases: list[str] | None = None
    ) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
            categories: Optional categories for the tool
            aliases: Optional aliases for the tool
            
        Raises:
            RegistryError: If tool name already registered
        """
        name = tool.metadata.name
        
        if name in self._tools:
            raise RegistryError(f"Tool already registered: {name}")
        
        self._tools[name] = tool
        
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(name)
        
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"Alias '{alias}' already exists, overwriting")
                self._aliases[alias] = name
        
        logger.info(f"Registered tool: {name}")
    
    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        categories: list[str] | None = None,
        timeout: float | None = None
    ) -> FunctionTool:
        """
        Register a function as a tool.
        
        Args:
            func: Function to register
            name: Tool name
            description: Tool description
            categories: Tool categories
            timeout: Execution timeout
            
        Returns:
            The created FunctionTool
        """
        tool = FunctionTool(
            func=func,
            name=name,
            description=description,
            timeout=timeout
        )
        self.register(tool, categories=categories)
        return tool
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered
        """
        if name not in self._tools:
            return False
        
        del self._tools[name]
        
        for category in self._categories:
            self._categories[category].discard(name)
        
        aliases_to_remove = [a for a, n in self._aliases.items() if n == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name or alias.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool instance or None
        """
        if name in self._tools:
            return self._tools[name]
        
        if name in self._aliases:
            return self._tools.get(self._aliases[name])
        
        return None
    
    def get_all(self) -> dict[str, BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of all tools
        """
        return self._tools.copy()
    
    def get_by_category(self, category: str) -> list[BaseTool]:
        """
        Get tools by category.
        
        Args:
            category: Category name
            
        Returns:
            List of tools in category
        """
        tool_names = self._categories.get(category, set())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_categories(self) -> list[str]:
        """
        Get all categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_metadata(self, name: str) -> ToolMetadata | None:
        """
        Get tool metadata by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolMetadata or None
        """
        tool = self.get(name)
        return tool.metadata if tool else None
    
    def get_all_metadata(self) -> list[ToolMetadata]:
        """
        Get metadata for all tools.
        
        Returns:
            List of all tool metadata
        """
        return [tool.metadata for tool in self._tools.values()]
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Get JSON schemas for all tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_tools_description(self) -> str:
        """
        Get formatted description of all tools.
        
        Returns:
            Formatted string describing all tools
        """
        lines = []
        
        for name, tool in sorted(self._tools.items()):
            meta = tool.metadata
            lines.append(f"\n### {name}")
            lines.append(f"Description: {meta.description}")
            
            if meta.parameters:
                lines.append("Parameters:")
                for param in meta.parameters:
                    required = " (required)" if param.required else ""
                    default = f" [default: {param.default}]" if param.default is not None else ""
                    lines.append(f"  - {param.name}: {param.type}{required}{default}")
                    if param.description:
                        lines.append(f"    {param.description}")
            
            lines.append(f"Returns: {meta.returns}")
        
        return "\n".join(lines)
    
    def has_tool(self, name: str) -> bool:
        """
        Check if tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool exists
        """
        return name in self._tools or name in self._aliases
    
    def list_tools(self) -> list[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def list_aliases(self) -> dict[str, str]:
        """
        List all aliases.
        
        Returns:
            Dictionary of alias -> tool name
        """
        return self._aliases.copy()
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        self._aliases.clear()
        logger.info("ToolRegistry cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with stats
        """
        total_executions = sum(t._execution_count for t in self._tools.values())
        total_errors = sum(t._error_count for t in self._tools.values())
        
        return {
            "total_tools": len(self._tools),
            "total_categories": len(self._categories),
            "total_aliases": len(self._aliases),
            "total_executions": total_executions,
            "total_errors": total_errors,
            "tools": {name: tool.get_stats() for name, tool in self._tools.items()}
        }
    
    def __contains__(self, name: str) -> bool:
        return self.has_tool(name)
    
    def __getitem__(self, name: str) -> BaseTool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name}")
        return tool
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.
    
    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    tool: BaseTool,
    categories: list[str] | None = None,
    aliases: list[str] | None = None
) -> None:
    """
    Register a tool in the global registry.
    
    Args:
        tool: Tool to register
        categories: Optional categories
        aliases: Optional aliases
    """
    get_registry().register(tool, categories, aliases)


def get_tool(name: str) -> BaseTool | None:
    """
    Get a tool from the global registry.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance or None
    """
    return get_registry().get(name)
