"""
Example tools for AI Agent Core System.
Demonstrates tool implementation patterns.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any

from app.tools.base import BaseTool, ToolMetadata, ToolParameter, tool

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Tool for basic arithmetic calculations."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Perform basic arithmetic calculations",
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')",
                    required=True
                )
            ],
            returns="number",
            examples=[
                {"expression": "2 + 3"},
                {"expression": "10 * 5 - 3"}
            ]
        )
    
    async def _execute(self, expression: str) -> float:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")


class DateTimeTool(BaseTool):
    """Tool for getting current date and time."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="datetime",
            description="Get current date and time in various formats",
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="format",
                    type="string",
                    description="Output format: 'iso', 'date', 'time', 'timestamp'",
                    required=False,
                    default="iso",
                    enum=["iso", "date", "time", "timestamp"]
                ),
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone name (e.g., 'UTC', 'America/New_York')",
                    required=False,
                    default="UTC"
                )
            ],
            returns="string"
        )
    
    async def _execute(self, format: str = "iso", timezone: str = "UTC") -> str:
        now = datetime.utcnow()
        
        if format == "iso":
            return now.isoformat()
        elif format == "date":
            return now.strftime("%Y-%m-%d")
        elif format == "time":
            return now.strftime("%H:%M:%S")
        elif format == "timestamp":
            return str(int(now.timestamp()))
        else:
            return now.isoformat()


class EchoTool(BaseTool):
    """Simple echo tool for testing."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="echo",
            description="Echo back the input message",
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Message to echo back",
                    required=True
                ),
                ToolParameter(
                    name="prefix",
                    type="string",
                    description="Optional prefix to add",
                    required=False,
                    default=""
                )
            ],
            returns="string"
        )
    
    async def _execute(self, message: str, prefix: str = "") -> str:
        result = f"{prefix}{message}" if prefix else message
        return result


class DelayTool(BaseTool):
    """Tool for introducing delays (useful for testing)."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="delay",
            description="Introduce a delay in seconds",
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="seconds",
                    type="number",
                    description="Number of seconds to delay",
                    required=True
                )
            ],
            returns="string",
            timeout=60.0
        )
    
    async def _execute(self, seconds: float) -> str:
        if seconds < 0:
            raise ValueError("Delay cannot be negative")
        if seconds > 60:
            raise ValueError("Maximum delay is 60 seconds")
        
        await asyncio.sleep(seconds)
        return f"Delayed for {seconds} seconds"


class TextProcessingTool(BaseTool):
    """Tool for text processing operations."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_process",
            description="Process text with various operations",
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="text",
                    type="string",
                    description="Text to process",
                    required=True
                ),
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    required=False,
                    default="lowercase",
                    enum=["lowercase", "uppercase", "reverse", "word_count", "char_count"]
                )
            ],
            returns="string or number"
        )
    
    async def _execute(self, text: str, operation: str = "lowercase") -> Any:
        if operation == "lowercase":
            return text.lower()
        elif operation == "uppercase":
            return text.upper()
        elif operation == "reverse":
            return text[::-1]
        elif operation == "word_count":
            return len(text.split())
        elif operation == "char_count":
            return len(text)
        else:
            return text


@tool(name="random_number", description="Generate a random number in range")
async def random_number(min_val: int = 0, max_val: int = 100) -> int:
    """
    Generate a random number between min_val and max_val.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        
    Returns:
        Random integer in range
    """
    import random
    return random.randint(min_val, max_val)


@tool(name="json_format", description="Format and validate JSON string")
def format_json(json_string: str, indent: int = 2) -> str:
    """
    Format a JSON string with proper indentation.
    
    Args:
        json_string: JSON string to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted JSON string
    """
    import json
    try:
        parsed = json.loads(json_string)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def register_example_tools() -> None:
    """Register all example tools."""
    from app.tools.registry import get_registry
    
    registry = get_registry()
    
    registry.register(CalculatorTool(), categories=["math", "utility"])
    registry.register(DateTimeTool(), categories=["utility", "time"])
    registry.register(EchoTool(), categories=["utility", "test"])
    registry.register(DelayTool(), categories=["utility", "test"])
    registry.register(TextProcessingTool(), categories=["text", "utility"])
    
    registry.register_function(
        random_number,
        categories=["math", "random"]
    )
    registry.register_function(
        format_json,
        categories=["utility", "text"]
    )
    
    logger.info(f"Registered {len(registry)} example tools")
