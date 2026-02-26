"""
Reasoning core module for AI Agent Core System.
Encapsulates vLLM interface for thought generation and action planning.
"""
import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.models.schemas import Message, MessageType, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningCoreError(Exception):
    """Exception raised for reasoning core errors."""
    pass


class ReasoningCore:
    """
    Core reasoning module that interfaces with vLLM for generation.
    
    Responsible for:
    - Generating thoughts and actions based on context
    - Managing conversation with the LLM
    - Parsing structured output from model responses
    """
    
    SYSTEM_PROMPT = """You are an intelligent AI agent with access to tools.
    
Your responses must be in JSON format with the following structure:
{
    "thought": "Your reasoning process and analysis",
    "action": "tool_name or 'final_answer'",
    "action_input": {"param1": "value1", ...},
    "confidence": 0.0-1.0,
    "requires_reflection": true/false
}

When you have a final answer, set action to "final_answer" and put the answer in action_input["answer"].

Always think step by step and explain your reasoning clearly.
Be precise and thorough in your analysis.
If you need more information, use available tools.
If you're confident in your answer, provide it as final_answer."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None
    ) -> None:
        """
        Initialize reasoning core with vLLM configuration.
        
        Args:
            base_url: vLLM API base URL (defaults to settings)
            api_key: vLLM API key (defaults to settings)
            model: Model name (defaults to settings)
            max_tokens: Maximum tokens for generation (defaults to settings)
            temperature: Generation temperature (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
        """
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.vllm_model
        self.max_tokens = max_tokens or settings.vllm_max_tokens
        self.temperature = temperature or settings.vllm_temperature
        self.timeout = timeout or settings.vllm_timeout
        
        self._client: httpx.AsyncClient | None = None
        logger.info(f"ReasoningCore initialized with model: {self.model}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("ReasoningCore HTTP client closed")
    
    def _build_messages(
        self,
        conversation_history: list[Message],
        context: dict[str, Any] | None = None,
        tools_description: str | None = None
    ) -> list[dict[str, str]]:
        """
        Build messages list for API request.
        
        Args:
            conversation_history: List of previous messages
            context: Additional context to include
            tools_description: Description of available tools
            
        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        if tools_description:
            messages.append({
                "role": "system",
                "content": f"Available tools:\n{tools_description}"
            })
        
        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            messages.append({
                "role": "system",
                "content": f"Additional context:\n{context_str}"
            })
        
        for msg in conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def _parse_response(self, content: str) -> ReasoningOutput:
        """
        Parse LLM response into structured output.
        
        Args:
            content: Raw response content
            
        Returns:
            ReasoningOutput: Parsed output
            
        Raises:
            ReasoningCoreError: If parsing fails
        """
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ReasoningCoreError("No JSON object found in response")
            
            json_str = content[json_start:json_end]
            data = json.loads(json_str)
            
            return ReasoningOutput(
                thought=data.get("thought", ""),
                action=data.get("action", "final_answer"),
                action_input=data.get("action_input", {}),
                confidence=float(data.get("confidence", 1.0)),
                requires_reflection=bool(data.get("requires_reflection", False))
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return ReasoningOutput(
                thought=content,
                action="final_answer",
                action_input={"answer": content},
                confidence=0.5,
                requires_reflection=True
            )
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise ReasoningCoreError(f"Failed to parse response: {e}")
    
    async def generate(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
        tools_description: str | None = None
    ) -> ReasoningOutput:
        """
        Generate reasoning output based on conversation history.
        
        Args:
            messages: Conversation history
            context: Additional context for generation
            tools_description: Description of available tools
            
        Returns:
            ReasoningOutput: Structured reasoning output
            
        Raises:
            ReasoningCoreError: If generation fails
        """
        client = await self._get_client()
        api_messages = self._build_messages(messages, context, tools_description)
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        try:
            logger.debug(f"Sending request to vLLM with {len(api_messages)} messages")
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            logger.debug(f"Received response: {content[:200]}...")
            return self._parse_response(content)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from vLLM: {e}")
            raise ReasoningCoreError(f"vLLM API error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error to vLLM: {e}")
            raise ReasoningCoreError(f"Failed to connect to vLLM: {e}")
        except KeyError as e:
            logger.error(f"Unexpected response format from vLLM: {e}")
            raise ReasoningCoreError(f"Invalid response format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}")
            raise ReasoningCoreError(f"Generation failed: {e}")
    
    async def generate_with_retry(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
        tools_description: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> ReasoningOutput:
        """
        Generate with automatic retry on failure.
        
        Args:
            messages: Conversation history
            context: Additional context
            tools_description: Available tools description
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            ReasoningOutput: Structured reasoning output
        """
        last_error: Exception | None = None
        
        for attempt in range(max_retries):
            try:
                return await self.generate(messages, context, tools_description)
            except ReasoningCoreError as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
        
        raise ReasoningCoreError(f"All {max_retries} attempts failed. Last error: {last_error}")
    
    async def reflect(
        self,
        messages: list[Message],
        action_taken: str,
        result: str,
        success: bool
    ) -> str:
        """
        Generate reflection on a completed action.
        
        Args:
            messages: Conversation history
            action_taken: Description of action taken
            result: Result of the action
            success: Whether action was successful
            
        Returns:
            str: Reflection text
        """
        reflection_prompt = f"""Reflect on the following action:
        
Action: {action_taken}
Result: {result}
Success: {success}

Provide a brief reflection on:
1. Was this the right action to take?
2. What could be improved?
3. What should be done next?

Keep your reflection concise and actionable."""
        
        reflection_messages = messages + [
            Message(role=MessageType.USER, content=reflection_prompt)
        ]
        
        try:
            output = await self.generate(reflection_messages)
            return output.thought
        except Exception as e:
            logger.error(f"Reflection generation failed: {e}")
            return f"Reflection failed: {str(e)}"
    
    def __repr__(self) -> str:
        return f"ReasoningCore(model={self.model}, base_url={self.base_url})"
