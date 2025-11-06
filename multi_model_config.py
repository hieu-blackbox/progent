"""
Multi-Model Configuration System for AgentDojo Experiments

This module provides configuration management for experiments using different models
for policy generation vs agent execution in the AgentDojo banking suite.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

from agentdojo.models import ModelsEnum


class ModelRole(Enum):
    """Defines the role of a model in the agent pipeline."""
    POLICY = "policy"
    EXECUTION = "execution"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model: ModelsEnum
    role: ModelRole
    description: str
    cost_tier: str  # "cheap", "medium", "expensive"


@dataclass
class ExperimentConfig:
    """Configuration for a multi-model experiment."""
    name: str
    description: str
    policy_model: ModelsEnum
    execution_model: ModelsEnum
    expected_cost_savings: Optional[float] = None
    expected_performance_impact: Optional[str] = None


class MultiModelConfigManager:
    """Manages multi-model experiment configurations."""
    
    def __init__(self):
        self.model_configs = self._initialize_model_configs()
        self.experiment_configs = self._initialize_experiment_configs()
    
    def _initialize_model_configs(self) -> Dict[ModelsEnum, ModelConfig]:
        """Initialize model configurations with cost tiers."""
        return {
            # Cheap models for policy generation
            ModelsEnum.GPT_3_5_TURBO_0125: ModelConfig(
                model=ModelsEnum.GPT_3_5_TURBO_0125,
                role=ModelRole.POLICY,
                description="GPT-3.5 Turbo - Fast and cost-effective for policy generation",
                cost_tier="cheap"
            ),
            ModelsEnum.GPT_4O_MINI_2024_07_18: ModelConfig(
                model=ModelsEnum.GPT_4O_MINI_2024_07_18,
                role=ModelRole.POLICY,
                description="GPT-4o Mini - Balanced performance and cost for policy generation",
                cost_tier="cheap"
            ),
            ModelsEnum.CLAUDE_3_HAIKU_20240307: ModelConfig(
                model=ModelsEnum.CLAUDE_3_HAIKU_20240307,
                role=ModelRole.POLICY,
                description="Claude 3 Haiku - Fast and efficient for policy generation",
                cost_tier="cheap"
            ),
            ModelsEnum.GEMINI_1_5_FLASH_002: ModelConfig(
                model=ModelsEnum.GEMINI_1_5_FLASH_002,
                role=ModelRole.POLICY,
                description="Gemini 1.5 Flash - Quick policy generation",
                cost_tier="cheap"
            ),
            
            # Expensive models for execution
            ModelsEnum.GPT_4O_2024_08_06: ModelConfig(
                model=ModelsEnum.GPT_4O_2024_08_06,
                role=ModelRole.EXECUTION,
                description="GPT-4o - High-quality agent execution",
                cost_tier="expensive"
            ),
            ModelsEnum.CLAUDE_3_5_SONNET_20241022: ModelConfig(
                model=ModelsEnum.CLAUDE_3_5_SONNET_20241022,
                role=ModelRole.EXECUTION,
                description="Claude 3.5 Sonnet - Advanced reasoning for execution",
                cost_tier="expensive"
            ),
            ModelsEnum.GEMINI_1_5_PRO_002: ModelConfig(
                model=ModelsEnum.GEMINI_1_5_PRO_002,
                role=ModelRole.EXECUTION,
                description="Gemini 1.5 Pro - Professional-grade execution",
                cost_tier="expensive"
            ),
        }
    
    def _initialize_experiment_configs(self) -> List[ExperimentConfig]:
        """Initialize predefined experiment configurations."""
        return [
            ExperimentConfig(
                name="baseline_gpt4o",
                description="Baseline: GPT-4o for both policy and execution",
                policy_model=ModelsEnum.GPT_4O_2024_08_06,
                execution_model=ModelsEnum.GPT_4O_2024_08_06,
                expected_cost_savings=0.0,
                expected_performance_impact="none"
            ),
            ExperimentConfig(
                name="cost_optimized_gpt",
                description="Cost-optimized: GPT-4o Mini for policy, GPT-4o for execution",
                policy_model=ModelsEnum.GPT_4O_MINI_2024_07_18,
                execution_model=ModelsEnum.GPT_4O_2024_08_06,
                expected_cost_savings=0.6,
                expected_performance_impact="minimal"
            ),
            ExperimentConfig(
                name="mixed_providers",
                description="Mixed: Claude Haiku for policy, GPT-4o for execution",
                policy_model=ModelsEnum.CLAUDE_3_HAIKU_20240307,
                execution_model=ModelsEnum.GPT_4O_2024_08_06,
                expected_cost_savings=0.7,
                expected_performance_impact="low"
            ),
            ExperimentConfig(
                name="ultra_cheap_policy",
                description="Ultra-cheap: GPT-3.5 for policy, GPT-4o for execution",
                policy_model=ModelsEnum.GPT_3_5_TURBO_0125,
                execution_model=ModelsEnum.GPT_4O_2024_08_06,
                expected_cost_savings=0.8,
                expected_performance_impact="moderate"
            ),
            ExperimentConfig(
                name="claude_premium",
                description="Claude premium: Claude Haiku for policy, Claude Sonnet for execution",
                policy_model=ModelsEnum.CLAUDE_3_HAIKU_20240307,
                execution_model=ModelsEnum.CLAUDE_3_5_SONNET_20241022,
                expected_cost_savings=0.5,
                expected_performance_impact="minimal"
            ),
            ExperimentConfig(
                name="gemini_mixed",
                description="Gemini mixed: Gemini Flash for policy, Gemini Pro for execution",
                policy_model=ModelsEnum.GEMINI_1_5_FLASH_002,
                execution_model=ModelsEnum.GEMINI_1_5_PRO_002,
                expected_cost_savings=0.4,
                expected_performance_impact="low"
            ),
        ]
    
    def get_experiment_config(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by name."""
        for config in self.experiment_configs:
            if config.name == name:
                return config
        return None
    
    def get_all_experiment_configs(self) -> List[ExperimentConfig]:
        """Get all available experiment configurations."""
        return self.experiment_configs
    
    def get_cheap_models(self) -> List[ModelsEnum]:
        """Get list of models suitable for policy generation (cheap)."""
        return [
            config.model for config in self.model_configs.values()
            if config.cost_tier == "cheap"
        ]
    
    def get_expensive_models(self) -> List[ModelsEnum]:
        """Get list of models suitable for agent execution (expensive)."""
        return [
            config.model for config in self.model_configs.values()
            if config.cost_tier == "expensive"
        ]
    
    def validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate that an experiment configuration is valid."""
        policy_valid = config.policy_model in self.get_cheap_models()
        execution_valid = config.execution_model in self.get_expensive_models()
        return policy_valid and execution_valid
    
    def create_custom_experiment(
        self,
        name: str,
        description: str,
        policy_model: ModelsEnum,
        execution_model: ModelsEnum
    ) -> ExperimentConfig:
        """Create a custom experiment configuration."""
        config = ExperimentConfig(
            name=name,
            description=description,
            policy_model=policy_model,
            execution_model=execution_model
        )
        
        if not self.validate_experiment_config(config):
            raise ValueError(f"Invalid experiment configuration: {name}")
        
        return config
    
    def save_experiment_results(
        self,
        experiment_name: str,
        results: Dict,
        output_dir: Path = Path("experiment_results")
    ) -> None:
        """Save experiment results to JSON file."""
        output_dir.mkdir(exist_ok=True)
        results_file = output_dir / f"{experiment_name}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_experiment_results(
        self,
        experiment_name: str,
        output_dir: Path = Path("experiment_results")
    ) -> Optional[Dict]:
        """Load experiment results from JSON file."""
        results_file = output_dir / f"{experiment_name}_results.json"
        
        if not results_file.exists():
            return None
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def print_experiment_summary(self) -> None:
        """Print a summary of all available experiments."""
        print("Available Multi-Model Experiments:")
        print("=" * 50)
        
        for config in self.experiment_configs:
            print(f"\nName: {config.name}")
            print(f"Description: {config.description}")
            print(f"Policy Model: {config.policy_model}")
            print(f"Execution Model: {config.execution_model}")
            if config.expected_cost_savings:
                print(f"Expected Cost Savings: {config.expected_cost_savings:.1%}")
            if config.expected_performance_impact:
                print(f"Expected Performance Impact: {config.expected_performance_impact}")


# Global instance for easy access
config_manager = MultiModelConfigManager()


def get_config_manager() -> MultiModelConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


if __name__ == "__main__":
    # Demo usage
    manager = get_config_manager()
    manager.print_experiment_summary()