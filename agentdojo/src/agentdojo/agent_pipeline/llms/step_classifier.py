"""Step classifier for ensemble LLM routing.

This module provides logic to classify agent steps into different categories
to route them to appropriate specialized models.
"""

from collections.abc import Sequence
from enum import Enum

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import ChatMessage


class StepType(Enum):
    """Types of steps that can be classified."""
    
    REASONING = "reasoning"
    """Steps requiring complex reasoning, planning, or decision-making."""
    
    CODING = "coding"
    """Steps involving tool calls with complex parameters or code generation."""
    
    SIMPLE = "simple"
    """Simple steps like formatting, basic responses, or straightforward operations."""


class StepClassifier:
    """Classifies agent steps to determine which model should handle them.
    
    The classifier uses heuristics based on:
    - Message history and context
    - Available tools and their complexity
    - Query patterns and keywords
    - Current conversation state
    """
    
    # Keywords that suggest reasoning tasks
    REASONING_KEYWORDS = [
        "plan", "strategy", "decide", "analyze", "evaluate", "consider",
        "think", "reason", "determine", "assess", "compare", "choose",
        "prioritize", "optimize", "solve", "figure out", "work out"
    ]
    
    # Keywords that suggest simple tasks
    SIMPLE_KEYWORDS = [
        "format", "display", "show", "list", "print", "output",
        "confirm", "acknowledge", "ok", "yes", "no", "done"
    ]
    
    def __init__(self):
        """Initialize the step classifier."""
        pass
    
    def classify(
        self,
        query: str,
        runtime: FunctionsRuntime,
        messages: Sequence[ChatMessage],
    ) -> StepType:
        """Classify the current step based on context.
        
        Args:
            query: The current query/task.
            runtime: The functions runtime with available tools.
            messages: The conversation history.
            
        Returns:
            The classified step type.
        """
        # Check if this is the initial planning phase
        if self._is_initial_planning(messages):
            return StepType.REASONING
        
        # Check if we need to make tool calls
        if self._requires_tool_calls(messages, runtime):
            # Determine if tool calls are complex
            if self._has_complex_tools(runtime):
                return StepType.CODING
            else:
                return StepType.SIMPLE
        
        # Check for reasoning keywords in query
        if self._contains_reasoning_keywords(query):
            return StepType.REASONING
        
        # Check for simple keywords
        if self._contains_simple_keywords(query):
            return StepType.SIMPLE
        
        # Check message complexity
        if self._is_complex_context(messages):
            return StepType.REASONING
        
        # Default to coding for tool-based interactions
        if len(runtime.functions) > 0:
            return StepType.CODING
        
        # Default to simple for basic responses
        return StepType.SIMPLE
    
    def _is_initial_planning(self, messages: Sequence[ChatMessage]) -> bool:
        """Check if this is the initial planning phase.
        
        Args:
            messages: The conversation history.
            
        Returns:
            True if this appears to be initial planning.
        """
        # If we have very few messages, it's likely initial planning
        if len(messages) <= 2:
            return True
        
        # If the last message is from user and no tools have been called yet
        if len(messages) > 0 and messages[-1]["role"] == "user":
            has_tool_calls = any(
                msg["role"] == "assistant" and msg.get("tool_calls")
                for msg in messages
            )
            if not has_tool_calls:
                return True
        
        return False
    
    def _requires_tool_calls(
        self,
        messages: Sequence[ChatMessage],
        runtime: FunctionsRuntime
    ) -> bool:
        """Check if the next step likely requires tool calls.
        
        Args:
            messages: The conversation history.
            runtime: The functions runtime.
            
        Returns:
            True if tool calls are likely needed.
        """
        # If no tools available, can't make tool calls
        if len(runtime.functions) == 0:
            return False
        
        # If last message is from user, we likely need to call tools
        if len(messages) > 0 and messages[-1]["role"] == "user":
            return True
        
        # If last message is a tool result, we might need more tool calls
        if len(messages) > 0 and messages[-1]["role"] == "tool":
            return True
        
        return False
    
    def _has_complex_tools(self, runtime: FunctionsRuntime) -> bool:
        """Check if available tools are complex.
        
        Args:
            runtime: The functions runtime.
            
        Returns:
            True if tools are complex.
        """
        # Consider tools complex if they have many parameters
        for func in runtime.functions.values():
            if func.parameters and len(func.parameters.model_fields) > 3:
                return True
        
        # If we have many tools, routing between them requires reasoning
        if len(runtime.functions) > 5:
            return True
        
        return False
    
    def _contains_reasoning_keywords(self, text: str) -> bool:
        """Check if text contains reasoning keywords.
        
        Args:
            text: The text to check.
            
        Returns:
            True if reasoning keywords are present.
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.REASONING_KEYWORDS)
    
    def _contains_simple_keywords(self, text: str) -> bool:
        """Check if text contains simple task keywords.
        
        Args:
            text: The text to check.
            
        Returns:
            True if simple keywords are present.
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.SIMPLE_KEYWORDS)
    
    def _is_complex_context(self, messages: Sequence[ChatMessage]) -> bool:
        """Check if the conversation context is complex.
        
        Args:
            messages: The conversation history.
            
        Returns:
            True if context is complex.
        """
        # Long conversations might need reasoning
        if len(messages) > 10:
            return True
        
        # Multiple tool calls suggest complexity
        tool_call_count = sum(
            1 for msg in messages
            if msg["role"] == "assistant" and msg.get("tool_calls")
        )
        if tool_call_count > 3:
            return True
        
        return False
