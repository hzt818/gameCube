"""
State machine module for AI Agent Core System.
Manages agent state transitions and validation.
"""
import logging
from typing import Callable

from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


class StateTransitionError(Exception):
    """Exception raised for invalid state transitions."""
    pass


class StateMachine:
    """
    Manages agent state transitions with validation.
    
    State flow:
    IDLE -> THINKING -> PLANNING -> EXECUTING -> REFLECTING -> COMPLETED
                           |              |              |
                           v              v              v
                         FAILED        FAILED        THINKING (loop)
    
    Any state can transition to FAILED or TIMEOUT.
    """
    
    VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
        AgentState.IDLE: {
            AgentState.THINKING,
            AgentState.FAILED,
        },
        AgentState.THINKING: {
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT,
        },
        AgentState.PLANNING: {
            AgentState.EXECUTING,
            AgentState.THINKING,
            AgentState.FAILED,
            AgentState.TIMEOUT,
        },
        AgentState.EXECUTING: {
            AgentState.REFLECTING,
            AgentState.THINKING,
            AgentState.PLANNING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT,
        },
        AgentState.REFLECTING: {
            AgentState.THINKING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT,
        },
        AgentState.COMPLETED: {
            AgentState.IDLE,
        },
        AgentState.FAILED: {
            AgentState.IDLE,
        },
        AgentState.TIMEOUT: {
            AgentState.IDLE,
        },
    }
    
    def __init__(self, initial_state: AgentState = AgentState.IDLE) -> None:
        """
        Initialize state machine with given initial state.
        
        Args:
            initial_state: Starting state for the machine
        """
        self._current_state = initial_state
        self._state_history: list[AgentState] = [initial_state]
        self._transition_callbacks: dict[AgentState, list[Callable[[AgentState, AgentState], None]]] = {}
        self._enter_callbacks: dict[AgentState, list[Callable[[], None]]] = {}
        self._exit_callbacks: dict[AgentState, list[Callable[[], None]]] = {}
        logger.info(f"State machine initialized with state: {initial_state.value}")
    
    @property
    def current_state(self) -> AgentState:
        """Get current state."""
        return self._current_state
    
    @property
    def state_history(self) -> list[AgentState]:
        """Get state transition history."""
        return self._state_history.copy()
    
    def can_transition_to(self, target_state: AgentState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: State to transition to
            
        Returns:
            bool: True if transition is valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in valid_targets
    
    def transition(self, target_state: AgentState) -> AgentState:
        """
        Transition to target state if valid.
        
        Args:
            target_state: State to transition to
            
        Returns:
            AgentState: New current state
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(target_state):
            valid_states = self.VALID_TRANSITIONS.get(self._current_state, set())
            raise StateTransitionError(
                f"Invalid transition from {self._current_state.value} to {target_state.value}. "
                f"Valid transitions: {[s.value for s in valid_states]}"
            )
        
        old_state = self._current_state
        self._execute_exit_callbacks(old_state)
        
        self._current_state = target_state
        self._state_history.append(target_state)
        
        self._execute_transition_callbacks(old_state, target_state)
        self._execute_enter_callbacks(target_state)
        
        logger.info(f"State transition: {old_state.value} -> {target_state.value}")
        return self._current_state
    
    def force_state(self, target_state: AgentState) -> AgentState:
        """
        Force transition to target state without validation.
        Use with caution - primarily for error recovery.
        
        Args:
            target_state: State to force transition to
            
        Returns:
            AgentState: New current state
        """
        old_state = self._current_state
        self._current_state = target_state
        self._state_history.append(target_state)
        logger.warning(f"Forced state transition: {old_state.value} -> {target_state.value}")
        return self._current_state
    
    def reset(self) -> AgentState:
        """
        Reset state machine to IDLE state.
        
        Returns:
            AgentState: IDLE state
        """
        self._current_state = AgentState.IDLE
        self._state_history = [AgentState.IDLE]
        logger.info("State machine reset to IDLE")
        return self._current_state
    
    def register_transition_callback(
        self,
        callback: Callable[[AgentState, AgentState], None]
    ) -> None:
        """
        Register callback for all state transitions.
        
        Args:
            callback: Function(old_state, new_state) called on transition
        """
        for state in AgentState:
            if state not in self._transition_callbacks:
                self._transition_callbacks[state] = []
            self._transition_callbacks[state].append(callback)
    
    def register_enter_callback(
        self,
        state: AgentState,
        callback: Callable[[], None]
    ) -> None:
        """
        Register callback for entering a specific state.
        
        Args:
            state: State to watch for
            callback: Function called when entering state
        """
        if state not in self._enter_callbacks:
            self._enter_callbacks[state] = []
        self._enter_callbacks[state].append(callback)
    
    def register_exit_callback(
        self,
        state: AgentState,
        callback: Callable[[], None]
    ) -> None:
        """
        Register callback for exiting a specific state.
        
        Args:
            state: State to watch for
            callback: Function called when exiting state
        """
        if state not in self._exit_callbacks:
            self._exit_callbacks[state] = []
        self._exit_callbacks[state].append(callback)
    
    def _execute_transition_callbacks(
        self,
        old_state: AgentState,
        new_state: AgentState
    ) -> None:
        """Execute all registered transition callbacks."""
        callbacks = self._transition_callbacks.get(new_state, [])
        for callback in callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in transition callback: {e}")
    
    def _execute_enter_callbacks(self, state: AgentState) -> None:
        """Execute all registered enter callbacks for state."""
        callbacks = self._enter_callbacks.get(state, [])
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in enter callback: {e}")
    
    def _execute_exit_callbacks(self, state: AgentState) -> None:
        """Execute all registered exit callbacks for state."""
        callbacks = self._exit_callbacks.get(state, [])
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in exit callback: {e}")
    
    def is_terminal_state(self) -> bool:
        """
        Check if current state is terminal (COMPLETED, FAILED, TIMEOUT).
        
        Returns:
            bool: True if in terminal state
        """
        return self._current_state in {
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT
        }
    
    def is_active_state(self) -> bool:
        """
        Check if current state is active (not IDLE or terminal).
        
        Returns:
            bool: True if in active state
        """
        return self._current_state not in {
            AgentState.IDLE,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.TIMEOUT
        }
    
    def __repr__(self) -> str:
        return f"StateMachine(current_state={self._current_state.value}, history_length={len(self._state_history)})"
