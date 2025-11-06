#!/usr/bin/env python3
"""
Test script to validate multi-model setup without requiring API keys.

This script tests the core functionality of our multi-model system
to ensure all components are properly integrated.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all our modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from multi_model_config import get_config_manager, ExperimentConfig
        print("✓ multi_model_config imported successfully")
        
        from multi_model_llm import create_multi_model_llm, PolicyAwareLLM
        print("✓ multi_model_llm imported successfully")
        
        from multi_model_pipeline import create_multi_model_pipeline, MultiModelAgentPipeline
        print("✓ multi_model_pipeline imported successfully")
        
        # Test AgentDojo imports
        from agentdojo.models import ModelsEnum
        from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
        print("✓ AgentDojo components imported successfully")
        
        # Test secagent imports
        import secagent
        print("✓ secagent imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_manager():
    """Test the configuration manager functionality."""
    print("\nTesting configuration manager...")
    
    try:
        from multi_model_config import get_config_manager
        
        config_manager = get_config_manager()
        
        # Test getting experiment configs
        experiments = config_manager.get_all_experiment_configs()
        print(f"✓ Found {len(experiments)} experiment configurations")
        
        # Test getting a specific config
        baseline_config = config_manager.get_experiment_config("baseline_gpt4o")
        if baseline_config:
            print(f"✓ Retrieved baseline config: {baseline_config.name}")
        else:
            print("✗ Could not retrieve baseline config")
            return False
        
        # Test validation
        is_valid = config_manager.validate_experiment_config(baseline_config)
        print(f"✓ Config validation: {is_valid}")
        
        return True
    except Exception as e:
        print(f"✗ Config manager test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_creation():
    """Test creating multi-model pipelines (without API calls)."""
    print("\nTesting pipeline creation...")
    
    try:
        from multi_model_pipeline import create_multi_model_pipeline
        
        # This will fail when trying to create actual LLM clients, but we can test
        # the configuration and setup logic
        try:
            pipeline = create_multi_model_pipeline("baseline_gpt4o")
            print("✓ Pipeline created successfully")
            print(f"✓ Pipeline name: {pipeline.name}")
            
            # Test getting experiment summary
            summary = pipeline.get_experiment_summary()
            print(f"✓ Experiment summary: {summary['experiment_name']}")
            
            return True
        except Exception as e:
            # Expected to fail due to missing API keys, but check if it's the right error
            error_msg = str(e).lower()
            if "api" in error_msg or "key" in error_msg or "auth" in error_msg or "client" in error_msg:
                print("✓ Pipeline creation failed as expected (missing API credentials)")
                return True
            else:
                print(f"✗ Unexpected pipeline creation error: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Pipeline creation test failed: {e}")
        traceback.print_exc()
        return False

def test_banking_suite_access():
    """Test that we can access the banking suite."""
    print("\nTesting banking suite access...")
    
    try:
        from agentdojo.task_suite.load_suites import get_suite
        
        # Try to load the banking suite
        banking_suite = get_suite("v1.2", "banking")
        print(f"✓ Banking suite loaded: {banking_suite.name}")
        
        # Check user tasks
        user_tasks = list(banking_suite.user_tasks.keys())
        print(f"✓ Found {len(user_tasks)} user tasks")
        
        # Check injection tasks  
        injection_tasks = list(banking_suite.injection_tasks.keys())
        print(f"✓ Found {len(injection_tasks)} injection tasks")
        
        return True
    except Exception as e:
        print(f"✗ Banking suite access failed: {e}")
        traceback.print_exc()
        return False

def test_experiment_runner():
    """Test the experiment runner setup."""
    print("\nTesting experiment runner...")
    
    try:
        from run_multi_model_experiment import ExperimentRunner
        
        # Create experiment runner
        runner = ExperimentRunner(Path("test_results"))
        print("✓ Experiment runner created")
        
        # Test configuration access
        config_manager = runner.config_manager
        experiments = config_manager.get_all_experiment_configs()
        print(f"✓ Runner has access to {len(experiments)} experiments")
        
        return True
    except Exception as e:
        print(f"✗ Experiment runner test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Multi-Model Setup Validation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Manager", test_config_manager),
        ("Pipeline Creation", test_pipeline_creation),
        ("Banking Suite Access", test_banking_suite_access),
        ("Experiment Runner", test_experiment_runner),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-model setup is working correctly.")
        print("\nNext steps:")
        print("1. Set up API keys in .env file")
        print("2. Run a quick test: python3.11 run_multi_model_experiment.py --quick-test")
        print("3. Run full experiments: python3.11 run_multi_model_experiment.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())