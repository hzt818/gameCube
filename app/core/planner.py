"""
Planner module for AI Agent Core System.
Handles task decomposition and planning.
"""
import logging
import re
from typing import Any

from app.config import settings
from app.models.schemas import AgentState, Message, MessageType, ReasoningOutput, TaskPlan

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Exception raised for planning errors."""
    pass


class Planner:
    """
    Task planning module that decomposes complex tasks into subtasks.
    
    Responsible for:
    - Analyzing task complexity
    - Decomposing tasks into atomic subtasks
    - Prioritizing and ordering tasks
    - Managing task dependencies
    """
    
    PLANNING_PROMPT = """You are a task planning expert. Analyze the given task and break it down into atomic subtasks.

For each subtask, provide:
1. A clear description
2. Dependencies on other subtasks (by ID)
3. Priority (0-10, higher = more important)
4. Estimated steps to complete

Output format (JSON array):
[
    {
        "description": "Clear description of subtask",
        "dependencies": ["task_id_1", "task_id_2"],
        "priority": 5,
        "estimated_steps": 2
    }
]

If the task is simple enough to complete in one step, return an empty array [].

Task to analyze: {task}

Context: {context}"""

    def __init__(self, max_subtasks: int = 10) -> None:
        """
        Initialize planner.
        
        Args:
            max_subtasks: Maximum number of subtasks to generate
        """
        self.max_subtasks = max_subtasks
        self._task_counter = 0
        logger.info(f"Planner initialized with max_subtasks={max_subtasks}")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter:04d}"
    
    def _parse_plan_response(self, content: str) -> list[dict[str, Any]]:
        """
        Parse planning response into task list.
        
        Args:
            content: Raw response content
            
        Returns:
            List of task dictionaries
        """
        try:
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            
            if json_start == -1:
                return []
            
            import json
            json_str = content[json_start:json_end]
            tasks = json.loads(json_str)
            
            if not isinstance(tasks, list):
                return []
            
            return tasks[:self.max_subtasks]
        except Exception as e:
            logger.error(f"Failed to parse plan response: {e}")
            return []
    
    async def plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        reasoning_core: Any = None
    ) -> list[TaskPlan]:
        """
        Create a plan for the given task.
        
        Args:
            task: Task description to plan
            context: Additional context for planning
            reasoning_core: Optional reasoning core for LLM-based planning
            
        Returns:
            List of TaskPlan objects
        """
        logger.info(f"Planning task: {task[:100]}...")
        
        if reasoning_core is None:
            return self._create_simple_plan(task)
        
        try:
            prompt = self.PLANNING_PROMPT.format(
                task=task,
                context=context or {}
            )
            
            messages = [Message(role=MessageType.USER, content=prompt)]
            output: ReasoningOutput = await reasoning_core.generate(messages)
            
            task_dicts = self._parse_plan_response(output.thought)
            
            if not task_dicts:
                return self._create_simple_plan(task)
            
            plans = []
            task_id_map: dict[int, str] = {}
            
            for idx, task_dict in enumerate(task_dicts):
                task_id = self._generate_task_id()
                task_id_map[idx] = task_id
                
                dependencies = []
                for dep_idx in task_dict.get("dependencies", []):
                    if isinstance(dep_idx, int) and dep_idx in task_id_map:
                        dependencies.append(task_id_map[dep_idx])
                
                plan = TaskPlan(
                    task_id=task_id,
                    description=task_dict.get("description", f"Subtask {idx + 1}"),
                    dependencies=dependencies,
                    priority=min(10, max(0, task_dict.get("priority", 5))),
                    status=AgentState.IDLE,
                    estimated_steps=max(1, task_dict.get("estimated_steps", 1))
                )
                plans.append(plan)
            
            plans.sort(key=lambda x: (-x.priority, len(x.dependencies)))
            
            logger.info(f"Created plan with {len(plans)} subtasks")
            return plans
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._create_simple_plan(task)
    
    def _create_simple_plan(self, task: str) -> list[TaskPlan]:
        """
        Create a simple single-task plan.
        
        Args:
            task: Task description
            
        Returns:
            List containing single TaskPlan
        """
        plan = TaskPlan(
            task_id=self._generate_task_id(),
            description=task,
            dependencies=[],
            priority=5,
            status=AgentState.IDLE,
            estimated_steps=1
        )
        logger.info("Created simple single-task plan")
        return [plan]
    
    def analyze_complexity(self, task: str) -> dict[str, Any]:
        """
        Analyze task complexity without full planning.
        
        Args:
            task: Task description
            
        Returns:
            Dictionary with complexity analysis
        """
        word_count = len(task.split())
        sentence_count = len(re.split(r'[.!?]+', task))
        
        complexity_keywords = [
            "and", "then", "after", "before", "while", "during",
            "first", "second", "finally", "next", "also", "additionally"
        ]
        
        task_lower = task.lower()
        keyword_count = sum(1 for kw in complexity_keywords if kw in task_lower)
        
        action_verbs = [
            "create", "build", "implement", "design", "develop",
            "analyze", "process", "generate", "compute", "calculate",
            "fetch", "retrieve", "store", "update", "delete"
        ]
        action_count = sum(1 for verb in action_verbs if verb in task_lower)
        
        complexity_score = min(10, (
            word_count / 20 +
            sentence_count * 0.5 +
            keyword_count * 0.5 +
            action_count * 1.5
        ))
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "keyword_count": keyword_count,
            "action_count": action_count,
            "complexity_score": round(complexity_score, 2),
            "is_complex": complexity_score > 3
        }
    
    def prioritize_tasks(self, tasks: list[TaskPlan]) -> list[TaskPlan]:
        """
        Re-prioritize tasks based on dependencies and importance.
        
        Args:
            tasks: List of tasks to prioritize
            
        Returns:
            Reordered list of tasks
        """
        if not tasks:
            return []
        
        completed_ids: set[str] = set()
        ordered: list[TaskPlan] = []
        remaining = list(tasks)
        
        while remaining:
            ready_tasks = [
                t for t in remaining
                if all(dep in completed_ids for dep in t.dependencies)
            ]
            
            if not ready_tasks:
                logger.warning("Circular dependency detected in tasks")
                ordered.extend(remaining)
                break
            
            ready_tasks.sort(key=lambda x: -x.priority)
            
            next_task = ready_tasks[0]
            ordered.append(next_task)
            completed_ids.add(next_task.task_id)
            remaining.remove(next_task)
        
        return ordered
    
    def get_ready_tasks(
        self,
        tasks: list[TaskPlan],
        completed_task_ids: set[str]
    ) -> list[TaskPlan]:
        """
        Get tasks that are ready to execute (all dependencies met).
        
        Args:
            tasks: All tasks
            completed_task_ids: Set of completed task IDs
            
        Returns:
            List of ready tasks
        """
        ready = []
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.status != AgentState.IDLE:
                continue
            if all(dep in completed_task_ids for dep in task.dependencies):
                ready.append(task)
        
        ready.sort(key=lambda x: -x.priority)
        return ready
    
    def reset(self) -> None:
        """Reset planner state."""
        self._task_counter = 0
        logger.info("Planner reset")
    
    def __repr__(self) -> str:
        return f"Planner(max_subtasks={self.max_subtasks}, task_counter={self._task_counter})"
