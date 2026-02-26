"""
Controller module for AI Agent Core System.
Orchestrates agent execution flow and manages component interactions.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Callable

from app.config import settings
from app.core.planner import Planner
from app.core.reasoning_core import ReasoningCore, ReasoningCoreError
from app.core.state_machine import StateMachine, StateTransitionError
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    AgentState,
    Message,
    MessageType,
    ReasoningOutput,
    SessionContext,
    TaskPlan,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ControllerError(Exception):
    """Exception raised for controller errors."""
    pass


class ExecutionLimitExceeded(ControllerError):
    """Raised when execution limits are exceeded."""
    pass


class Controller:
    """
    Main controller that orchestrates agent execution.
    
    Responsible for:
    - Managing execution flow
    - Coordinating between components
    - Enforcing limits and timeouts
    - Handling errors and recovery
    - Managing session state
    """
    
    def __init__(
        self,
        reasoning_core: ReasoningCore | None = None,
        planner: Planner | None = None,
        tool_router: Any = None,
        memory_manager: Any = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        tool_timeout: float | None = None,
        reflection_enabled: bool | None = None
    ) -> None:
        """
        Initialize controller with dependencies.
        
        Args:
            reasoning_core: Reasoning module for generation
            planner: Task planning module
            tool_router: Tool execution router
            memory_manager: Memory management module
            max_iterations: Maximum iterations per session
            max_tool_calls: Maximum tool calls per session
            tool_timeout: Tool execution timeout
            reflection_enabled: Whether to enable self-reflection
        """
        self.reasoning_core = reasoning_core or ReasoningCore()
        self.planner = planner or Planner()
        self.tool_router = tool_router
        self.memory_manager = memory_manager
        
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.max_tool_calls = max_tool_calls or settings.agent_max_tool_calls
        self.tool_timeout = tool_timeout or settings.agent_tool_timeout
        self.reflection_enabled = reflection_enabled if reflection_enabled is not None else settings.agent_reflection_enabled
        
        self._state_machine = StateMachine()
        self._sessions: dict[str, SessionContext] = {}
        self._hooks: dict[str, list[Callable]] = {
            "pre_iteration": [],
            "post_iteration": [],
            "pre_tool_call": [],
            "post_tool_call": [],
            "on_error": [],
            "on_complete": [],
        }
        
        logger.info(
            f"Controller initialized: max_iterations={self.max_iterations}, "
            f"max_tool_calls={self.max_tool_calls}, reflection={self.reflection_enabled}"
        )
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """
        Register a callback hook.
        
        Args:
            hook_name: Name of hook point
            callback: Callback function
        """
        if hook_name in self._hooks:
            self._hooks[hook_name].append(callback)
        else:
            raise ControllerError(f"Unknown hook: {hook_name}")
    
    async def _execute_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute all registered hooks for a point."""
        for callback in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback error: {e}")
    
    def _create_session(self, request: AgentRequest) -> SessionContext:
        """Create new session context."""
        session = SessionContext(
            session_id=request.session_id or SessionContext.model_fields["session_id"].default_factory(),
            user_id=request.user_id,
            conversation_history=[
                Message(role=MessageType.USER, content=request.query)
            ],
            metadata=request.context
        )
        self._sessions[session.session_id] = session
        logger.info(f"Created session: {session.session_id}")
        return session
    
    def _get_session(self, session_id: str) -> SessionContext | None:
        """Get existing session."""
        return self._sessions.get(session_id)
    
    def _check_limits(self, session: SessionContext) -> None:
        """Check if execution limits are exceeded."""
        if session.iteration_count >= self.max_iterations:
            raise ExecutionLimitExceeded(
                f"Maximum iterations ({self.max_iterations}) exceeded"
            )
        if session.total_tool_calls >= self.max_tool_calls:
            raise ExecutionLimitExceeded(
                f"Maximum tool calls ({self.max_tool_calls}) exceeded"
            )
    
    async def _execute_tool(self, tool_call: ToolCall, session: SessionContext) -> ToolResult:
        """
        Execute a tool call with timeout and error handling.
        
        Args:
            tool_call: Tool call to execute
            session: Current session context
            
        Returns:
            ToolResult: Execution result
        """
        start_time = datetime.utcnow()
        
        await self._execute_hooks("pre_tool_call", tool_call, session)
        
        if self.tool_router is None:
            result = ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status="failed",
                result=None,
                error="Tool router not configured"
            )
        else:
            try:
                result = await asyncio.wait_for(
                    self.tool_router.execute(tool_call),
                    timeout=self.tool_timeout
                )
            except asyncio.TimeoutError:
                result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    status="timeout",
                    result=None,
                    error=f"Tool execution timed out after {self.tool_timeout}s"
                )
            except Exception as e:
                result = ToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    status="failed",
                    result=None,
                    error=str(e)
                )
        
        end_time = datetime.utcnow()
        result.execution_time = (end_time - start_time).total_seconds()
        
        session.tool_calls.append(tool_call)
        session.tool_results.append(result)
        session.total_tool_calls += 1
        
        await self._execute_hooks("post_tool_call", tool_call, result, session)
        
        logger.info(
            f"Tool {tool_call.tool_name} executed: status={result.status}, "
            f"time={result.execution_time:.2f}s"
        )
        
        return result
    
    async def _run_iteration(self, session: SessionContext) -> ReasoningOutput:
        """
        Run a single iteration of the agent loop.
        
        Args:
            session: Current session context
            
        Returns:
            ReasoningOutput: Generated reasoning output
        """
        self._check_limits(session)
        
        session.iteration_count += 1
        session.updated_at = datetime.utcnow()
        
        await self._execute_hooks("pre_iteration", session)
        
        tools_desc = None
        if self.tool_router:
            tools_desc = self.tool_router.get_tools_description()
        
        output = await self.reasoning_core.generate_with_retry(
            messages=session.conversation_history,
            context=session.metadata,
            tools_description=tools_desc
        )
        
        thought_message = Message(
            role=MessageType.ASSISTANT,
            content=f"[Thought] {output.thought}"
        )
        session.conversation_history.append(thought_message)
        
        await self._execute_hooks("post_iteration", output, session)
        
        return output
    
    async def _handle_action(
        self,
        output: ReasoningOutput,
        session: SessionContext
    ) -> tuple[bool, str]:
        """
        Handle the action from reasoning output.
        
        Args:
            output: Reasoning output with action
            session: Current session
            
        Returns:
            Tuple of (is_final, result_text)
        """
        if output.action == "final_answer":
            answer = output.action_input.get("answer", str(output.action_input))
            return True, answer
        
        tool_call = ToolCall(
            tool_name=output.action,
            arguments=output.action_input
        )
        
        result = await self._execute_tool(tool_call, session)
        
        result_message = Message(
            role=MessageType.TOOL,
            content=f"[Tool Result] {result.result or result.error}"
        )
        session.conversation_history.append(result_message)
        
        return False, str(result.result) if result.result else result.error or "No result"
    
    async def _reflect(
        self,
        session: SessionContext,
        action: str,
        result: str,
        success: bool
    ) -> None:
        """
        Run reflection on completed action.
        
        Args:
            session: Current session
            action: Action taken
            result: Result of action
            success: Whether action succeeded
        """
        if not self.reflection_enabled:
            return
        
        try:
            self._state_machine.transition(AgentState.REFLECTING)
            
            reflection = await self.reasoning_core.reflect(
                messages=session.conversation_history,
                action_taken=action,
                result=result,
                success=success
            )
            
            reflection_message = Message(
                role=MessageType.ASSISTANT,
                content=f"[Reflection] {reflection}"
            )
            session.conversation_history.append(reflection_message)
            
            logger.debug(f"Reflection completed: {reflection[:100]}...")
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute agent request.
        
        Args:
            request: Agent request to process
            
        Returns:
            AgentResponse: Execution result
        """
        session = None
        thought_process: list[str] = []
        
        try:
            session = self._create_session(request)
            self._state_machine.reset()
            self._state_machine.transition(AgentState.THINKING)
            
            while not self._state_machine.is_terminal_state():
                self._check_limits(session)
                
                output = await self._run_iteration(session)
                thought_process.append(output.thought)
                
                is_final, result = await self._handle_action(output, session)
                
                if is_final:
                    self._state_machine.transition(AgentState.COMPLETED)
                    
                    return AgentResponse(
                        session_id=session.session_id,
                        response=result,
                        thought_process=thought_process,
                        tool_calls_made=session.total_tool_calls,
                        iterations_used=session.iteration_count,
                        status=AgentState.COMPLETED,
                        success=True
                    )
                
                success = "error" not in result.lower()
                await self._reflect(session, output.action, result, success)
                
                self._state_machine.transition(AgentState.THINKING)
            
            return AgentResponse(
                session_id=session.session_id,
                response="Execution terminated",
                thought_process=thought_process,
                tool_calls_made=session.total_tool_calls,
                iterations_used=session.iteration_count,
                status=self._state_machine.current_state,
                success=False,
                error="Unexpected termination"
            )
            
        except ExecutionLimitExceeded as e:
            logger.warning(f"Execution limits exceeded: {e}")
            if session:
                self._state_machine.force_state(AgentState.TIMEOUT)
            
            return AgentResponse(
                session_id=session.session_id if session else "unknown",
                response="",
                thought_process=thought_process,
                tool_calls_made=session.total_tool_calls if session else 0,
                iterations_used=session.iteration_count if session else 0,
                status=AgentState.TIMEOUT,
                success=False,
                error=str(e)
            )
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            await self._execute_hooks("on_error", e, session)
            
            if session:
                self._state_machine.force_state(AgentState.FAILED)
            
            return AgentResponse(
                session_id=session.session_id if session else "unknown",
                response="",
                thought_process=thought_process,
                tool_calls_made=session.total_tool_calls if session else 0,
                iterations_used=session.iteration_count if session else 0,
                status=AgentState.FAILED,
                success=False,
                error=str(e)
            )
        
        finally:
            await self._execute_hooks("on_complete", session)
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self.reasoning_core.close()
        logger.info("Controller shutdown complete")
    
    def __repr__(self) -> str:
        return (
            f"Controller(state={self._state_machine.current_state.value}, "
            f"sessions={len(self._sessions)})"
        )
