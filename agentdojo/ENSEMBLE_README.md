# Ensemble Model Implementation for AgentDojo

This implementation adds an ensemble approach to AgentDojo experiments, where different types of tasks are routed to specialized models optimized for those tasks.

## Overview

The ensemble approach uses three types of models:
- **Reasoning Models** (o3-mini, GPT-4.1): For planning, complex decision-making, and strategic thinking
- **Coding Models** (GPT-4o): For tool calls, code generation, and complex parameter handling
- **Small Models** (GPT-4o-mini): For simple formatting, basic responses, and straightforward operations

## Architecture

### Components

1. **StepClassifier** (`src/agentdojo/agent_pipeline/llms/step_classifier.py`)
   - Analyzes the current task context
   - Classifies steps into: REASONING, CODING, or SIMPLE
   - Uses heuristics based on:
     - Message history and conversation state
     - Available tools and their complexity
     - Query patterns and keywords
     - Current phase (initial planning vs. execution)

2. **EnsembleLLM** (`src/agentdojo/agent_pipeline/llms/ensemble_llm.py`)
   - Wraps multiple LLM instances
   - Routes requests to appropriate model based on classification
   - Tracks usage statistics per model type
   - Logs routing decisions for analysis

3. **Model Configurations** (`src/agentdojo/models.py`)
   - `ensemble-o3-gpt4-mini`: Uses o3-mini for reasoning
   - `ensemble-gpt41-gpt4-mini`: Uses GPT-4.1 for reasoning
   - Both use GPT-4o for coding and GPT-4o-mini for simple tasks

4. **Pipeline Integration** (`src/agentdojo/agent_pipeline/agent_pipeline.py`)
   - Automatic ensemble initialization
   - Seamless integration with existing pipeline architecture
   - Support for all existing defenses and configurations

## Usage

### Quick Start

1. **Install dependencies:**
   ```bash
   cd agentdojo
   pip install -e .
   cd ..
   pip install -e .  # Install progent
   ```

2. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # Add other API keys as needed
   ```

3. **Run ensemble experiments:**
   ```bash
   cd agentdojo
   ./run_ensemble.sh
   ```

### Running Specific Suites

To run a specific suite with ensemble:

```bash
cd agentdojo
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --logdir logs/ensemble
```

### Running with Attacks

```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --attack important_instructions \
  --logdir logs/ensemble
```

### Available Ensemble Models

- `ensemble-o3-gpt4-mini`: o3-mini + GPT-4o + GPT-4o-mini
- `ensemble-gpt41-gpt4-mini`: GPT-4.1 + GPT-4o + GPT-4o-mini

## Configuration

### Customizing Ensemble Behavior

You can customize the step classifier by modifying the heuristics in `step_classifier.py`:

```python
# Adjust reasoning keywords
REASONING_KEYWORDS = [
    "plan", "strategy", "decide", "analyze", ...
]

# Adjust simple task keywords
SIMPLE_KEYWORDS = [
    "format", "display", "show", "list", ...
]
```

### Adding New Ensemble Configurations

1. Add to `models.py`:
   ```python
   class ModelsEnum(StrEnum):
       ENSEMBLE_CUSTOM = "ensemble-custom"
   
   MODEL_PROVIDERS = {
       ModelsEnum.ENSEMBLE_CUSTOM: "ensemble",
   }
   
   ENSEMBLE_CONFIGS = {
       "ensemble-custom": ("reasoning-model", "coding-model", "simple-model"),
   }
   ```

2. Use in experiments:
   ```bash
   python -m agentdojo.scripts.benchmark \
     -s banking \
     --model ensemble-custom \
     --logdir logs/custom
   ```

## Monitoring and Analysis

### Viewing Routing Decisions

The ensemble logs routing decisions to stderr:

```
[Ensemble] Step 1: Routing to reasoning model (gpt-4.1-2025-04-14)
[Ensemble] Step 2: Routing to coding model (gpt-4o-2024-08-06)
[Ensemble] Step 3: Routing to simple model (gpt-4o-mini-2024-07-18)
```

### Usage Statistics

Every 5 steps, the ensemble logs usage statistics:

```
[Ensemble] Usage statistics (total: 10):
  Reasoning: 3 (30.0%)
  Coding: 5 (50.0%)
  Simple: 2 (20.0%)
```

### Token Usage

Token usage is tracked separately for each model:

```
[Agent] tokens (completion, prompt): 150, 500
total (completion, prompt): 1500, 5000
```

## Classification Logic

### Step Types

1. **REASONING**
   - Initial planning phase (first 1-2 messages)
   - Complex decision-making
   - Strategic thinking
   - Contains reasoning keywords: "plan", "strategy", "analyze", etc.
   - Long conversation contexts (>10 messages)
   - Multiple tool calls (>3)

2. **CODING**
   - Tool calls with complex parameters (>3 parameters)
   - Many available tools (>5)
   - Tool execution phase
   - Default for tool-based interactions

3. **SIMPLE**
   - Formatting and display tasks
   - Basic responses
   - Simple keywords: "format", "show", "list", etc.
   - No tools available
   - Straightforward operations

### Classification Flow

```
1. Check if initial planning → REASONING
2. Check if tool calls needed:
   - Complex tools → CODING
   - Simple tools → SIMPLE
3. Check for reasoning keywords → REASONING
4. Check for simple keywords → SIMPLE
5. Check context complexity → REASONING
6. Default: CODING (if tools) or SIMPLE (no tools)
```

## Performance Considerations

### Token Efficiency

The ensemble approach can be more token-efficient by:
- Using smaller models for simple tasks
- Reserving expensive reasoning models for complex decisions
- Optimizing model selection based on task requirements

### Latency

- Reasoning models (o3-mini, GPT-4.1) may have higher latency
- Coding models (GPT-4o) provide balanced performance
- Simple models (GPT-4o-mini) offer fastest responses

### Cost Optimization

- Reasoning models: Higher cost, used selectively
- Coding models: Medium cost, primary workhorse
- Simple models: Lower cost, frequent use

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
cd agentdojo
pip install -e .
```

### Model Not Found

Ensure the model is in `ModelsEnum` and `MODEL_PROVIDERS`:
```python
ModelsEnum.ENSEMBLE_GPT41_GPT4_MINI = "ensemble-gpt41-gpt4-mini"
MODEL_PROVIDERS[ModelsEnum.ENSEMBLE_GPT41_GPT4_MINI] = "ensemble"
```

### API Key Issues

Set all required API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # If using Claude
```

### Routing Issues

Check the logs for routing decisions:
```bash
tail -f logs/ensemble/*.log | grep "\[Ensemble\]"
```

## Examples

### Example 1: Banking Suite with Ensemble

```bash
cd agentdojo
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --logdir logs/banking-ensemble
```

### Example 2: All Suites with Attacks

```bash
cd agentdojo
./run_ensemble.sh
```

This runs all suites (banking, slack, travel, workspace) with and without attacks.

### Example 3: Single Task Testing

```bash
cd agentdojo
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --user-task task_0 \
  --logdir logs/test
```

## Future Enhancements

Potential improvements to the ensemble approach:

1. **Adaptive Classification**: Learn from past routing decisions
2. **Custom Classifiers**: Per-suite or per-task classifiers
3. **Dynamic Model Selection**: Choose models based on performance metrics
4. **Cost-Aware Routing**: Optimize for cost vs. performance trade-offs
5. **Parallel Execution**: Run multiple models and select best response

## References

- AgentDojo: https://github.com/ethz-spylab/agentdojo
- Progent Paper: https://arxiv.org/abs/2504.11703

## Support

For issues or questions:
1. Check the logs in `logs/ensemble/`
2. Run verification: `python3 verify_ensemble.py`
3. Review classification logic in `step_classifier.py`
4. Check model configurations in `models.py`
