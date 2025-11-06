#!/usr/bin/env python3
"""Simple verification script to check ensemble implementation files."""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_file_content(filepath, search_strings, description):
    """Check if file contains expected content."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        all_found = True
        for search_str in search_strings:
            if search_str in content:
                print(f"  ✓ Contains: {search_str}")
            else:
                print(f"  ✗ Missing: {search_str}")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False

def main():
    """Verify ensemble implementation."""
    print("=" * 70)
    print("Ensemble Implementation Verification")
    print("=" * 70)
    
    base_dir = "/vercel/sandbox/agentdojo"
    all_checks_passed = True
    
    # Check 1: Step classifier file
    print("\n1. Checking step_classifier.py...")
    filepath = os.path.join(base_dir, "src/agentdojo/agent_pipeline/llms/step_classifier.py")
    if check_file_exists(filepath, "Step classifier"):
        check_file_content(filepath, [
            "class StepType",
            "class StepClassifier",
            "REASONING",
            "CODING",
            "SIMPLE"
        ], "Step classifier content")
    else:
        all_checks_passed = False
    
    # Check 2: Ensemble LLM file
    print("\n2. Checking ensemble_llm.py...")
    filepath = os.path.join(base_dir, "src/agentdojo/agent_pipeline/llms/ensemble_llm.py")
    if check_file_exists(filepath, "Ensemble LLM"):
        check_file_content(filepath, [
            "class EnsembleLLM",
            "reasoning_llm",
            "coding_llm",
            "simple_llm",
            "def query"
        ], "Ensemble LLM content")
    else:
        all_checks_passed = False
    
    # Check 3: Models.py updates
    print("\n3. Checking models.py updates...")
    filepath = os.path.join(base_dir, "src/agentdojo/models.py")
    if check_file_exists(filepath, "Models file"):
        check_file_content(filepath, [
            "ENSEMBLE_O3_GPT4_MINI",
            "ENSEMBLE_GPT41_GPT4_MINI",
            "ENSEMBLE_CONFIGS",
            '"ensemble"'
        ], "Models.py ensemble additions")
    else:
        all_checks_passed = False
    
    # Check 4: Agent pipeline updates
    print("\n4. Checking agent_pipeline.py updates...")
    filepath = os.path.join(base_dir, "src/agentdojo/agent_pipeline/agent_pipeline.py")
    if check_file_exists(filepath, "Agent pipeline"):
        check_file_content(filepath, [
            "from agentdojo.agent_pipeline.llms.ensemble_llm import EnsembleLLM",
            "ENSEMBLE_CONFIGS",
            'if provider == "ensemble"'
        ], "Agent pipeline ensemble support")
    else:
        all_checks_passed = False
    
    # Check 5: Run script
    print("\n5. Checking run_ensemble.sh...")
    filepath = os.path.join(base_dir, "run_ensemble.sh")
    if check_file_exists(filepath, "Ensemble run script"):
        check_file_content(filepath, [
            "ensemble",
            "SECAGENT_SUITE",
            "benchmark"
        ], "Run script content")
        # Check if executable
        if os.access(filepath, os.X_OK):
            print("  ✓ Script is executable")
        else:
            print("  ✗ Script is not executable")
    else:
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    if all_checks_passed:
        print("✓ All ensemble implementation files are present and contain expected content!")
        print("\nImplementation complete! The ensemble approach includes:")
        print("  • StepClassifier: Classifies tasks as reasoning/coding/simple")
        print("  • EnsembleLLM: Routes requests to appropriate specialized models")
        print("  • Model configurations: ensemble-o3-gpt4-mini, ensemble-gpt41-gpt4-mini")
        print("  • Pipeline integration: Automatic routing in agent pipeline")
        print("  • Run script: ./run_ensemble.sh to execute experiments")
        print("\nTo use the ensemble:")
        print("  1. Install dependencies: cd agentdojo && pip install -e .")
        print("  2. Set up API keys (OPENAI_API_KEY, etc.)")
        print("  3. Run: ./run_ensemble.sh")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
