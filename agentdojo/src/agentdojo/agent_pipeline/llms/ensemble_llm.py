"""Ensemble LLM that routes requests to specialized models.

This module provides an ensemble approach where different types of tasks
are routed to specialized models optimized for those tasks.
"""

import logging
import sys
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.step_classifier import StepClassifier, StepType
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


class EnsembleLLM(BasePipelineElement):
    """Ensemble LLM that routes to specialized models based on task type.
    
    This class wraps multiple LLM instances and intelligently routes requests
    to the most appropriate model based on the task characteristics:
    - Reasoning models for planning and complex decisions
    - Coding models for tool calls and code generation
    - Small models for simple formatting and responses
    
    Args:
        reasoning_llm: LLM for reasoning and planning tasks.
        coding_llm: LLM for coding and tool execution tasks.
        simple_llm: LLM for simple formatting and response tasks.
        classifier: Optional custom step classifier. If None, uses default.
        name: Optional name for this ensemble.
    """
    
    def __init__(
        self,
        reasoning_llm: BasePipelineElement,
        coding_llm: BasePipelineElement,
        simple_llm: BasePipelineElement,
        classifier: StepClassifier | None = None,
        name: str | None = None,
    ) -> None:
        self.reasoning_llm = reasoning_llm
        self.coding_llm = coding_llm
        self.simple_llm = simple_llm
        self.classifier = classifier or StepClassifier()
        self.name = name or "ensemble"
        
        # Track usage statistics
        self.usage_stats = {
            StepType.REASONING: 0,
            StepType.CODING: 0,
            StepType.SIMPLE: 0,
        }
        
        # Get model names for logging
        self.reasoning_name = getattr(reasoning_llm, "model", "reasoning")
        self.coding_name = getattr(coding_llm, "model", "coding")
        self.simple_name = getattr(simple_llm, "model", "simple")
        
        logging.info(
            f"Initialized EnsembleLLM with:\n"
            f"  Reasoning: {self.reasoning_name}\n"
            f"  Coding: {self.coding_name}\n"
            f"  Simple: {self.simple_name}"
        )
    
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """Execute the query by routing to the appropriate model.
        
        Args:
            query: The query to execute.
            runtime: The functions runtime.
            env: The environment.
            messages: The conversation history.
            extra_args: Extra arguments.
            
        Returns:
            Tuple of (query, runtime, env, messages, extra_args).
        """
        # Classify the step
        step_type = self.classifier.classify(query, runtime, messages)
        
        # Update usage statistics
        self.usage_stats[step_type] += 1
        
        # Select the appropriate model
        if step_type == StepType.REASONING:
            selected_llm = self.reasoning_llm
            model_name = self.reasoning_name
        elif step_type == StepType.CODING:
            selected_llm = self.coding_llm
            model_name = self.coding_name
        else:  # StepType.SIMPLE
            selected_llm = self.simple_llm
            model_name = self.simple_name
        
        # Log the routing decision
        print(
            f"[Ensemble] Step {sum(self.usage_stats.values())}: "
            f"Routing to {step_type.value} model ({model_name})",
            file=sys.stderr
        )
        
        # Execute with the selected model
        result = selected_llm.query(query, runtime, env, messages, extra_args)
        
        # Log usage statistics periodically
        if sum(self.usage_stats.values()) % 5 == 0:
            self._log_usage_stats()
        
        return result
    
    def _log_usage_stats(self) -> None:
        """Log usage statistics for the ensemble."""
        total = sum(self.usage_stats.values())
        if total == 0:
            return
        
        print(
            f"[Ensemble] Usage statistics (total: {total}):\n"
            f"  Reasoning: {self.usage_stats[StepType.REASONING]} "
            f"({self.usage_stats[StepType.REASONING] / total * 100:.1f}%)\n"
            f"  Coding: {self.usage_stats[StepType.CODING]} "
            f"({self.usage_stats[StepType.CODING] / total * 100:.1f}%)\n"
            f"  Simple: {self.usage_stats[StepType.SIMPLE]} "
            f"({self.usage_stats[StepType.SIMPLE] / total * 100:.1f}%)",
            file=sys.stderr
        )
    
    def get_usage_stats(self) -> dict[StepType, int]:
        """Get the current usage statistics.
        
        Returns:
            Dictionary mapping step types to usage counts.
        """
        return self.usage_stats.copy()
    
    def reset_usage_stats(self) -> None:
        """Reset the usage statistics."""
        self.usage_stats = {
            StepType.REASONING: 0,
            StepType.CODING: 0,
            StepType.SIMPLE: 0,
        }


def create_ensemble_llm(
    reasoning_model: str,
    coding_model: str,
    simple_model: str,
    reasoning_llm_factory,
    coding_llm_factory,
    simple_llm_factory,
    name: str | None = None,
) -> EnsembleLLM:
    """Factory function to create an ensemble LLM.
    
    Args:
        reasoning_model: Model name for reasoning tasks.
        coding_model: Model name for coding tasks.
        simple_model: Model name for simple tasks.
        reasoning_llm_factory: Factory function to create reasoning LLM.
        coding_llm_factory: Factory function to create coding LLM.
        simple_llm_factory: Factory function to create simple LLM.
        name: Optional name for the ensemble.
        
    Returns:
        Configured EnsembleLLM instance.
    """
    reasoning_llm = reasoning_llm_factory(reasoning_model)
    coding_llm = coding_llm_factory(coding_model)
    simple_llm = simple_llm_factory(simple_model)
    
    ensemble_name = name or f"ensemble-{reasoning_model}-{coding_model}-{simple_model}"
    
    return EnsembleLLM(
        reasoning_llm=reasoning_llm,
        coding_llm=coding_llm,
        simple_llm=simple_llm,
        name=ensemble_name,
    )
