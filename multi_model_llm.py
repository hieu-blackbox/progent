"""
Multi-Model LLM Wrapper for AgentDojo Experiments

This module provides LLM wrappers that route different operations to different models,
enabling cost optimization by using cheaper models for policy generation and expensive
models for agent execution.
"""

import os
import sys
from typing import Sequence, Dict, Any, Optional
from collections.abc import Sequence as ABCSequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.agent_pipeline import get_llm
from agentdojo.functions_runtime import FunctionsRuntime, Env, EmptyEnv
from agentdojo.models import ModelsEnum, MODEL_PROVIDERS
from agentdojo.types import ChatMessage
from multi_model_config import ExperimentConfig, get_config_manager


class MultiModelLLM(BasePipelineElement):
    """
    LLM wrapper that uses different models for policy generation vs execution.
    
    This class routes policy-related operations to a cheaper model while using
    an expensive model for actual agent execution tasks.
    """
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        self.policy_llm = get_llm(
            MODEL_PROVIDERS[experiment_config.policy_model],
            experiment_config.policy_model
        )
        self.execution_llm = get_llm(
            MODEL_PROVIDERS[experiment_config.execution_model], 
            experiment_config.execution_model
        )
        
        # Set name for logging
        self.name = f"multi-{experiment_config.policy_model}-{experiment_config.execution_model}"
        
        # Track token usage for cost analysis
        self.policy_tokens = {"prompt": 0, "completion": 0}
        self.execution_tokens = {"prompt": 0, "completion": 0}
        
        print(f"[MultiModelLLM] Initialized with policy model: {experiment_config.policy_model}, "
              f"execution model: {experiment_config.execution_model}", file=sys.stderr)
    
    def _is_policy_generation_context(self, messages: Sequence[ChatMessage]) -> bool:
        """
        Determine if the current context is for policy generation.
        
        This heuristic looks for policy-related keywords in the messages to decide
        which model to use.
        """
        if not messages:
            return False
        
        # Check the last few messages for policy-related content
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        
        policy_keywords = [
            "policy", "restriction", "permission", "tool", "security",
            "TOOLS:", "USER_QUERY:", "restrictions", "allowed", "blocked",
            "parameter", "schema", "validation"
        ]
        
        for message in recent_messages:
            content = message.content.lower() if hasattr(message, 'content') else str(message).lower()
            if any(keyword in content for keyword in policy_keywords):
                return True
        
        return False
    
    def _update_token_usage(self, is_policy: bool, prompt_tokens: int, completion_tokens: int):
        """Update token usage tracking for cost analysis."""
        if is_policy:
            self.policy_tokens["prompt"] += prompt_tokens
            self.policy_tokens["completion"] += completion_tokens
        else:
            self.execution_tokens["prompt"] += prompt_tokens
            self.execution_tokens["completion"] += completion_tokens
    
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: ABCSequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, ABCSequence[ChatMessage], dict]:
        """
        Route the query to the appropriate model based on context.
        """
        # Determine which model to use based on context
        is_policy_context = self._is_policy_generation_context(messages)
        
        if is_policy_context:
            print(f"[MultiModelLLM] Using policy model: {self.experiment_config.policy_model}", 
                  file=sys.stderr)
            selected_llm = self.policy_llm
        else:
            print(f"[MultiModelLLM] Using execution model: {self.experiment_config.execution_model}", 
                  file=sys.stderr)
            selected_llm = self.execution_llm
        
        # Execute the query with the selected model
        try:
            result = selected_llm.query(query, runtime, env, messages, extra_args)
            
            # TODO: Extract token usage from the result if available
            # This would require modifications to the underlying LLM classes
            # For now, we'll track this at a higher level
            
            return result
        except Exception as e:
            print(f"[MultiModelLLM] Error with {'policy' if is_policy_context else 'execution'} "
                  f"model: {e}", file=sys.stderr)
            raise
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage for cost analysis."""
        return {
            "experiment_config": self.experiment_config.name,
            "policy_model": self.experiment_config.policy_model,
            "execution_model": self.experiment_config.execution_model,
            "policy_tokens": self.policy_tokens,
            "execution_tokens": self.execution_tokens,
            "total_policy_tokens": sum(self.policy_tokens.values()),
            "total_execution_tokens": sum(self.execution_tokens.values()),
        }


class PolicyAwareLLM(BasePipelineElement):
    """
    Alternative implementation that explicitly handles policy generation.
    
    This version integrates more directly with the secagent policy system
    to ensure policy generation always uses the cheaper model.
    """
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        self.policy_llm = get_llm(
            MODEL_PROVIDERS[experiment_config.policy_model],
            experiment_config.policy_model
        )
        self.execution_llm = get_llm(
            MODEL_PROVIDERS[experiment_config.execution_model],
            experiment_config.execution_model
        )
        
        self.name = f"policy-aware-{experiment_config.policy_model}-{experiment_config.execution_model}"
        
        # Override the policy model in secagent
        self._override_policy_model()
        
        print(f"[PolicyAwareLLM] Initialized with policy override", file=sys.stderr)
    
    def _override_policy_model(self):
        """Override the policy model in secagent to use our cheaper model."""
        # This requires modifying the secagent module to use our policy model
        # We'll set the environment variable that secagent uses
        os.environ["SECAGENT_POLICY_MODEL"] = str(self.experiment_config.policy_model)
        
        # Also try to directly modify the secagent module if possible
        try:
            import secagent.tool as secagent_tool
            secagent_tool.policy_model = str(self.experiment_config.policy_model)
            print(f"[PolicyAwareLLM] Set secagent policy model to: {self.experiment_config.policy_model}", 
                  file=sys.stderr)
        except Exception as e:
            print(f"[PolicyAwareLLM] Could not directly set secagent policy model: {e}", 
                  file=sys.stderr)
    
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: ABCSequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, ABCSequence[ChatMessage], dict]:
        """
        Always use the execution model for agent queries.
        Policy generation is handled separately by secagent.
        """
        print(f"[PolicyAwareLLM] Using execution model: {self.experiment_config.execution_model}", 
              file=sys.stderr)
        
        return self.execution_llm.query(query, runtime, env, messages, extra_args)


def create_multi_model_llm(experiment_name: str) -> BasePipelineElement:
    """
    Factory function to create a multi-model LLM based on experiment configuration.
    """
    config_manager = get_config_manager()
    experiment_config = config_manager.get_experiment_config(experiment_name)
    
    if experiment_config is None:
        raise ValueError(f"Unknown experiment configuration: {experiment_name}")
    
    # For now, use PolicyAwareLLM as it integrates better with secagent
    return PolicyAwareLLM(experiment_config)


def list_available_experiments() -> None:
    """List all available multi-model experiments."""
    config_manager = get_config_manager()
    config_manager.print_experiment_summary()


if __name__ == "__main__":
    # Demo usage
    print("Available Multi-Model Experiments:")
    list_available_experiments()
    
    # Test creating a multi-model LLM
    try:
        llm = create_multi_model_llm("cost_optimized_gpt")
        print(f"\nCreated multi-model LLM: {llm.name}")
    except Exception as e:
        print(f"Error creating multi-model LLM: {e}")