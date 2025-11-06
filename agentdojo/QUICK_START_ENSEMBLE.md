# Quick Start: Ensemble Models

## What is it?

An intelligent routing system that uses different models for different types of tasks:
- 🧠 **Reasoning models** (o3-mini, GPT-4.1) → Planning & complex decisions
- 💻 **Coding models** (GPT-4o) → Tool calls & code generation
- ⚡ **Small models** (GPT-4o-mini) → Simple formatting & responses

## Setup (3 steps)

### 1. Install Dependencies
```bash
cd agentdojo
pip install -e .
cd ..
pip install -e .
```

### 2. Set API Key
```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. Run Experiments
```bash
cd agentdojo
./run_ensemble.sh
```

## Usage Examples

### Run All Suites
```bash
./run_ensemble.sh
```

### Run Single Suite
```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --logdir logs/ensemble
```

### Run with Attacks
```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --attack important_instructions \
  --logdir logs/ensemble
```

### Run Single Task (Testing)
```bash
python -m agentdojo.scripts.benchmark \
  -s banking \
  --model ensemble-gpt41-gpt4-mini \
  --user-task task_0 \
  --logdir logs/test
```

## Available Models

| Model | Reasoning | Coding | Simple |
|-------|-----------|--------|--------|
| `ensemble-o3-gpt4-mini` | o3-mini | GPT-4o | GPT-4o-mini |
| `ensemble-gpt41-gpt4-mini` | GPT-4.1 | GPT-4o | GPT-4o-mini |

## Monitoring

### View Logs
```bash
tail -f logs/ensemble/*.log
```

### View Routing Decisions
```bash
tail -f logs/ensemble/*.log | grep "\[Ensemble\]"
```

### Check Progress
```bash
ls -lh logs/ensemble/
```

## Verification

Check if everything is set up correctly:
```bash
python3 verify_ensemble.py
```

Expected: `✓ All ensemble implementation files are present and contain expected content!`

## How It Works

```
User Query
    ↓
Classify Task Type
    ↓
┌───┴───┬───────┬────────┐
↓       ↓       ↓        ↓
Planning Tool   Simple   Response
         Call
↓       ↓       ↓        ↓
o3-mini GPT-4o  GPT-4o   Result
/GPT-4.1        -mini
```

## Task Classification

- **Reasoning**: Initial planning, complex decisions, strategy
- **Coding**: Tool calls, complex parameters, code generation
- **Simple**: Formatting, display, basic responses

## Troubleshooting

### Import Error
```bash
cd agentdojo && pip install -e .
```

### API Key Error
```bash
export OPENAI_API_KEY="your-key"
```

### Model Not Found
Check that you're using:
- `ensemble-o3-gpt4-mini` or
- `ensemble-gpt41-gpt4-mini`

### Check Logs
```bash
cat logs/ensemble/*.log | grep -i error
```

## Stop Running Experiments

```bash
pkill -f 'agentdojo.scripts.benchmark'
```

## Files & Documentation

- 📖 Full docs: `ENSEMBLE_README.md`
- 📋 Summary: `../ENSEMBLE_IMPLEMENTATION_SUMMARY.md`
- ✅ Verify: `verify_ensemble.py`
- 🚀 Run: `run_ensemble.sh`

## Key Features

✅ Automatic task classification  
✅ Intelligent model routing  
✅ Usage statistics tracking  
✅ Cost optimization  
✅ Full logging & monitoring  
✅ Easy configuration  

## Performance

- **Cost**: Optimized by using small models for simple tasks
- **Speed**: Fast responses with GPT-4o-mini for basic operations
- **Quality**: Best model for each task type

## Next Steps

1. Run verification: `python3 verify_ensemble.py`
2. Test with single task: `--user-task task_0`
3. Run full suite: `./run_ensemble.sh`
4. Monitor logs: `tail -f logs/ensemble/*.log`
5. Analyze results: Check `logs/ensemble/` directory

## Support

Need help? Check:
1. Logs: `logs/ensemble/*.log`
2. Verification: `python3 verify_ensemble.py`
3. Full docs: `ENSEMBLE_README.md`
