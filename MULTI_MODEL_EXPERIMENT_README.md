# Multi-Model Agent Pipeline Experiment

This repository implements a multi-model experiment system for the AgentDojo banking suite, allowing you to use different models for policy generation vs agent execution to optimize cost and performance.

## Overview

The experiment system enables you to:
- Use **cheaper models** (GPT-4o Mini, Claude Haiku, GPT-3.5) for **policy generation**
- Use **expensive models** (GPT-4o, Claude Sonnet, Gemini Pro) for **agent execution**
- Compare cost savings vs performance trade-offs
- Evaluate security and utility across different model combinations

## Architecture

### Core Components

1. **Multi-Model Configuration System** (`multi_model_config.py`)
   - Defines experiment configurations with different model combinations
   - Manages cost tiers and expected performance impacts
   - Provides validation and result storage

2. **Multi-Model LLM Wrapper** (`multi_model_llm.py`)
   - Routes policy generation to cheaper models
   - Routes agent execution to expensive models
   - Integrates with secagent policy system

3. **Multi-Model Pipeline** (`multi_model_pipeline.py`)
   - Extends AgentDojo's pipeline system
   - Supports all existing defenses (tool_filter, pi_detector, etc.)
   - Maintains compatibility with existing benchmarks

4. **Benchmark Integration** (`multi_model_benchmark.py`)
   - Extended benchmark script for multi-model experiments
   - Tracks token usage and cost metrics
   - Supports all AgentDojo features (attacks, defenses, etc.)

5. **Experiment Runner** (`run_multi_model_experiment.py`)
   - Orchestrates multiple experiments
   - Generates comparative analysis
   - Produces detailed reports

## Available Experiments

| Experiment Name | Policy Model | Execution Model | Expected Cost Savings | Performance Impact |
|----------------|--------------|-----------------|----------------------|-------------------|
| `baseline_gpt4o` | GPT-4o | GPT-4o | 0% | None |
| `cost_optimized_gpt` | GPT-4o Mini | GPT-4o | 60% | Minimal |
| `mixed_providers` | Claude Haiku | GPT-4o | 70% | Low |
| `ultra_cheap_policy` | GPT-3.5 | GPT-4o | 80% | Moderate |
| `claude_premium` | Claude Haiku | Claude Sonnet | 50% | Minimal |
| `gemini_mixed` | Gemini Flash | Gemini Pro | 40% | Low |

## Installation

1. **Install Python 3.11+** (required for AgentDojo)
```bash
# On Amazon Linux 2023
sudo dnf install -y python3.11 python3.11-pip
```

2. **Install Dependencies**
```bash
# Install secagent
cd /vercel/sandbox
python3.11 -m pip install -e .

# Install agentdojo
cd agentdojo
python3.11 -m pip install -e .
```

3. **Validate Setup**
```bash
cd /vercel/sandbox
python3.11 test_multi_model_setup.py
```

## Usage

### List Available Experiments
```bash
python3.11 multi_model_benchmark.py --list-experiments
```

### Run a Single Experiment
```bash
# Run cost-optimized experiment on banking suite
python3.11 multi_model_benchmark.py \
    --experiment cost_optimized_gpt \
    --suite banking \
    --save-results

# Run with attack
python3.11 multi_model_benchmark.py \
    --experiment mixed_providers \
    --suite banking \
    --attack important_instructions \
    --save-results
```

### Run Experiment Suite
```bash
# Quick test with minimal experiments
python3.11 run_multi_model_experiment.py --quick-test

# Full experiment suite
python3.11 run_multi_model_experiment.py

# Custom experiment selection
python3.11 run_multi_model_experiment.py \
    --experiments baseline_gpt4o cost_optimized_gpt \
    --attacks important_instructions \
    --output-dir my_results
```

## Configuration

### Environment Variables
```bash
# Set policy model for secagent (automatically handled by multi-model system)
export SECAGENT_POLICY_MODEL="gpt-4o-mini-2024-07-18"

# Enable/disable policy generation
export SECAGENT_GENERATE="True"
```

### API Keys
Create a `.env` file with your API keys:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google Cloud (for Gemini)
GCP_PROJECT=your_project
GCP_LOCATION=your_location
```

## Banking Suite Details

The AgentDojo banking suite includes:

### User Tasks (16 total)
- **Bill Payment**: Pay bills from files, follow landlord instructions
- **Transaction Analysis**: Calculate spending, find specific transactions
- **Refund Processing**: Calculate and process refunds
- **Subscription Management**: Set up recurring payments
- **Account Management**: Update addresses, passwords, rent payments

### Injection Tasks (9 total)
- Extract user information (music services, food preferences, phone models)
- Modify recurring payments to attacker accounts
- Transfer money to attacker accounts
- Exfiltrate scheduled transaction data

### Available Tools
- `send_money()` - Transfer funds
- `get_balance()` - Check account balance
- `get_most_recent_transactions()` - View transaction history
- `schedule_transaction()` - Set up future payments
- `update_scheduled_transaction()` - Modify scheduled payments
- `read_file()` - Read bills and notices
- `update_user_info()` - Change account details

## Results and Analysis

### Metrics Tracked
- **Utility**: Task completion success rate
- **Security**: Attack resistance effectiveness  
- **Cost**: Token usage and estimated API costs
- **Performance**: Execution time and latency

### Expected Outcomes
1. **Cost Savings**: 40-80% reduction in API costs depending on experiment
2. **Performance**: Minimal to moderate impact on task completion
3. **Security**: Policy generation quality may affect security policies
4. **Efficiency**: Faster policy generation with cheaper models

### Result Files
- `experiment_results/` - Detailed JSON results for each experiment
- `experiment_report.txt` - Human-readable summary report
- `logs/` - Detailed execution logs for debugging

## Extending the System

### Adding New Experiments
```python
from multi_model_config import get_config_manager

config_manager = get_config_manager()
custom_config = config_manager.create_custom_experiment(
    name="my_experiment",
    description="Custom experiment description",
    policy_model=ModelsEnum.GPT_3_5_TURBO_0125,
    execution_model=ModelsEnum.CLAUDE_3_5_SONNET_20241022
)
```

### Adding New Models
Update `multi_model_config.py` to include new models in the appropriate cost tiers.

### Custom Analysis
Results are stored in JSON format for easy analysis:
```python
import json
from pathlib import Path

# Load experiment results
with open("experiment_results/cost_optimized_gpt_results.json") as f:
    results = json.load(f)

# Analyze utility scores
utility_scores = results["results"]["banking"]["utility_results"]
avg_utility = sum(utility_scores.values()) / len(utility_scores)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both secagent and agentdojo are installed with Python 3.11+
2. **API Key Errors**: Check that all required API keys are set in `.env`
3. **Model Not Found**: Verify model names match those in `agentdojo.models.ModelsEnum`
4. **Permission Errors**: Ensure write permissions for result directories

### Debug Mode
```bash
# Enable verbose logging
export SECAGENT_GENERATE="True"
export SECAGENT_IGNORE_UPDATE_ERROR="False"

# Run with debug output
python3.11 multi_model_benchmark.py \
    --experiment baseline_gpt4o \
    --suite banking \
    --user-task user_task_0 \
    --save-results
```

### Validation
```bash
# Test setup without API calls
python3.11 test_multi_model_setup.py

# Validate specific experiment
python3.11 -c "
from multi_model_config import get_config_manager
config = get_config_manager().get_experiment_config('cost_optimized_gpt')
print(f'Valid: {get_config_manager().validate_experiment_config(config)}')
"
```

## Research Applications

This system enables research into:
- **Cost-Performance Trade-offs**: Quantify savings vs utility loss
- **Model Specialization**: Optimal model selection for different tasks
- **Security Policy Quality**: Impact of cheaper models on security policies
- **Hybrid AI Systems**: Multi-model architectures for complex tasks

## Contributing

To contribute new experiments or improvements:
1. Add new experiment configurations in `multi_model_config.py`
2. Test with `test_multi_model_setup.py`
3. Run validation experiments
4. Update documentation

## License

This experiment system extends the AgentDojo and Progent frameworks. Please refer to their respective licenses for usage terms.