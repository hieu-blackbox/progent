#!/bin/bash
# Script to run ensemble experiments with AgentDojo
# This uses different models for different types of tasks:
# - Reasoning models (o3-mini, GPT-4.1) for planning and complex decisions
# - Coding models (GPT-4o) for tool calls and code generation
# - Small models (GPT-4o-mini) for simple formatting and responses

log_dir="logs/ensemble"

mkdir -p $log_dir

# Choose ensemble configuration
# Options: ensemble-o3-gpt4-mini, ensemble-gpt41-gpt4-mini
model="ensemble-gpt41-gpt4-mini"

# Policy model for security policy generation
export SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06"

# Use manual policies (set to "True" for auto-generated policies)
export SECAGENT_GENERATE="False"

# Uncomment for auto policy updates
# export SECAGENT_UPDATE="True"
# export SECAGENT_IGNORE_UPDATE_ERROR="True"

export COLUMNS=300

echo "Running ensemble experiments with model: $model"
echo "Logs will be saved to: $log_dir"
echo ""
echo "Ensemble configuration:"
echo "  - Reasoning: o3-mini or GPT-4.1 (for planning and complex decisions)"
echo "  - Coding: GPT-4o (for tool calls and code generation)"
echo "  - Simple: GPT-4o-mini (for formatting and basic responses)"
echo ""

# Run benchmarks without attacks
echo "Starting benchmarks without attacks..."
SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --logdir $log_dir > $log_dir/banking-no-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --logdir $log_dir > $log_dir/slack-no-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --logdir $log_dir > $log_dir/travel-no-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --logdir $log_dir > $log_dir/workspace-no-attack.log 2>&1 &

# Run benchmarks with attacks
echo "Starting benchmarks with attacks..."
SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --attack important_instructions --logdir $log_dir > $log_dir/banking-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --attack important_instructions --logdir $log_dir > $log_dir/slack-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --attack important_instructions --logdir $log_dir > $log_dir/travel-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --attack important_instructions --logdir $log_dir > $log_dir/workspace-attack.log 2>&1 &

echo ""
echo "All benchmarks started in background!"
echo "Monitor progress with: tail -f $log_dir/*.log"
echo ""
echo "To check running processes: ps aux | grep benchmark"
echo "To stop all benchmarks: pkill -f 'agentdojo.scripts.benchmark'"
