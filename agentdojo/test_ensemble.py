#!/usr/bin/env python3
"""Test script to verify ensemble LLM implementation."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.models import ModelsEnum


def test_ensemble_initialization():
    """Test that ensemble models can be initialized."""
    print("Testing ensemble initialization...")
    
    try:
        # Test ensemble-gpt41-gpt4-mini configuration
        config = PipelineConfig(
            llm=ModelsEnum.ENSEMBLE_GPT41_GPT4_MINI,
            defense=None,
            system_message_name=None,
            system_message="You are a helpful assistant.",
        )
        
        pipeline = AgentPipeline.from_config(config)
        print(f"✓ Successfully created pipeline with ensemble model")
        print(f"  Pipeline name: {pipeline.name}")
        
        # Check that the pipeline has the expected structure
        if len(pipeline.elements) > 0:
            print(f"  Pipeline has {len(pipeline.elements)} elements")
            for i, element in enumerate(pipeline.elements):
                print(f"    Element {i}: {type(element).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_models():
    """Test both ensemble configurations."""
    print("\nTesting ensemble model configurations...")
    
    configs_to_test = [
        ModelsEnum.ENSEMBLE_GPT41_GPT4_MINI,
        ModelsEnum.ENSEMBLE_O3_GPT4_MINI,
    ]
    
    results = {}
    for model_enum in configs_to_test:
        print(f"\nTesting {model_enum}...")
        try:
            config = PipelineConfig(
                llm=model_enum,
                defense=None,
                system_message_name=None,
                system_message="You are a helpful assistant.",
            )
            pipeline = AgentPipeline.from_config(config)
            print(f"  ✓ {model_enum} initialized successfully")
            results[model_enum] = True
        except Exception as e:
            print(f"  ✗ {model_enum} failed: {e}")
            results[model_enum] = False
    
    return all(results.values())


def main():
    """Run all tests."""
    print("=" * 60)
    print("Ensemble LLM Implementation Test")
    print("=" * 60)
    
    # Test 1: Basic initialization
    test1_passed = test_ensemble_initialization()
    
    # Test 2: All ensemble configurations
    test2_passed = test_ensemble_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Basic initialization: {'PASS' if test1_passed else 'FAIL'}")
    print(f"All configurations: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        print("\nYou can now run ensemble experiments with:")
        print("  cd agentdojo && ./run_ensemble.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
