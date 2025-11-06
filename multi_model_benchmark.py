"""
Multi-Model Benchmark Script for AgentDojo

This script extends the AgentDojo benchmark to support multi-model experiments
where policy generation uses cheaper models and agent execution uses expensive models.
"""

import importlib
import logging
import warnings
import time
import json
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import click
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

# Import AgentDojo components
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite

# Import our multi-model components
from multi_model_pipeline import create_multi_model_pipeline
from multi_model_config import get_config_manager
from secagent import reset_security_policy


def benchmark_suite_multi_model(
    experiment_name: str,
    suite: TaskSuite,
    logdir: Path,
    force_rerun: bool,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    attack: str | None = None,
    defense: str | None = None,
    system_message_name: str | None = None,
    system_message: str | None = None,
    live: Live | None = None,
) -> Tuple[SuiteResults, dict]:
    """
    Run benchmark with multi-model pipeline and return results plus cost analysis.
    """
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    print(f"Running multi-model benchmark for suite: '{suite.name}'")
    print(f"Using experiment: '{experiment_name}'")
    if attack is not None:
        print(f"Using attack: '{attack}'")
    if defense is not None:
        print(f"Using defense: '{defense}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")

    # Create multi-model pipeline
    pipeline = create_multi_model_pipeline(
        experiment_name=experiment_name,
        defense=defense,
        system_message_name=system_message_name,
        system_message=system_message
    )
    
    # Track experiment start time
    start_time = time.time()
    
    with OutputLogger(str(logdir), live=live):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
            )
        else:
            attacker_ = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
            )
    
    # Calculate experiment metrics
    end_time = time.time()
    experiment_metrics = {
        "experiment_name": experiment_name,
        "suite_name": suite.name,
        "duration_seconds": end_time - start_time,
        "pipeline_summary": pipeline.get_experiment_summary(),
        "attack": attack,
        "defense": defense,
        "user_tasks_count": len(user_tasks) if user_tasks else len(suite.user_tasks),
        "injection_tasks_count": len(injection_tasks) if injection_tasks else len(suite.injection_tasks),
    }
    
    # Try to get token usage if available
    if hasattr(pipeline.multi_model_llm, 'get_token_usage_summary'):
        experiment_metrics["token_usage"] = pipeline.multi_model_llm.get_token_usage_summary()
    
    print(f"Finished multi-model benchmark for suite: '{suite.name}'")
    
    return results, experiment_metrics


def show_multi_model_results(
    experiment_name: str, 
    suite_name: str, 
    results: SuiteResults, 
    metrics: dict,
    show_security_results: bool
):
    """Show results with multi-model specific information."""
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results) if utility_results else 0

    print(f"\nResults for Multi-Model Experiment: {experiment_name}")
    print(f"Suite: {suite_name}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
    print(f"Average utility: {avg_utility * 100:.2f}%")
    
    # Show model configuration
    pipeline_summary = metrics.get("pipeline_summary", {})
    print(f"Policy Model: {pipeline_summary.get('policy_model', 'Unknown')}")
    print(f"Execution Model: {pipeline_summary.get('execution_model', 'Unknown')}")
    if pipeline_summary.get('expected_cost_savings'):
        print(f"Expected Cost Savings: {pipeline_summary['expected_cost_savings']:.1%}")

    if show_security_results:
        passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
        total_injection_tasks = len(results["injection_tasks_utility_results"])
        print(f"Passed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

        security_results = results["security_results"].values()
        avg_security = sum(security_results) / len(security_results) if security_results else 0
        print(f"Average security: {avg_security * 100:.2f}%")
    
    # Show token usage if available
    token_usage = metrics.get("token_usage")
    if token_usage:
        print(f"\nToken Usage:")
        print(f"  Policy tokens: {token_usage.get('total_policy_tokens', 0)}")
        print(f"  Execution tokens: {token_usage.get('total_execution_tokens', 0)}")


@click.command()
@click.option(
    "--experiment",
    "-e",
    type=str,
    required=False,
    help="Multi-model experiment name to run (e.g., 'cost_optimized_gpt', 'mixed_providers')",
)
@click.option(
    "--benchmark-version",
    default="v1.2",
    type=str,
    help="The version of the benchmark to run. Defaults to `v1.2`.",
)
@click.option(
    "--logdir",
    default="./multi_model_runs",
    type=Path,
    help="The directory to save logs. Defaults to `./multi_model_runs`.",
)
@click.option(
    "--attack",
    type=str,
    default=None,
    help=f"The attack to use. `None` by default. It should be one of {ATTACKS}. If `None`, no attack is used.",
)
@click.option(
    "--defense",
    type=click.Choice(DEFENSES),
    default=None,
    help="The defense to use. `None` by default.",
)
@click.option(
    "--system-message-name",
    type=str,
    default=None,
    help="The name of the system message to use among the default ones in `data/system_messages.yaml`.",
)
@click.option(
    "--system-message",
    type=str,
    default=None,
    help="The system message to use (as a string). If provided, `--system-message-name` is ignored.",
)
@click.option(
    "--user-task",
    "-ut",
    "user_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--injection-task",
    "-it",
    "injection_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The injection tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--suite",
    "-s",
    "suites",
    type=str,
    multiple=True,
    default=("banking",),  # Default to banking suite for our experiment
    help="The suites to benchmark. Defaults to banking suite.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="How many suites can be benchmarked in parallel. Nothing is parallelized by default.",
)
@click.option(
    "--force-rerun",
    "-f",
    is_flag=True,
    help="Whether to re-run tasks that have already been run.",
)
@click.option(
    "--save-results",
    is_flag=True,
    help="Whether to save detailed results to JSON files.",
)
@click.option(
    "--list-experiments",
    is_flag=True,
    help="List available multi-model experiments and exit.",
)
def main(
    experiment: str | None,
    suites: tuple[str, ...],
    benchmark_version: str = "v1.2",
    logdir: Path = Path("./multi_model_runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    attack: str | None = None,
    defense: str | None = None,
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    save_results: bool = False,
    list_experiments: bool = False,
):
    """Run multi-model experiments on AgentDojo banking suite."""
    
    if list_experiments:
        config_manager = get_config_manager()
        config_manager.print_experiment_summary()
        return
    
    if experiment is None:
        print("Error: --experiment is required when not listing experiments")
        print("Use --list-experiments to see available experiments")
        return
    
    # Validate experiment name
    config_manager = get_config_manager()
    if config_manager.get_experiment_config(experiment) is None:
        print(f"Error: Unknown experiment '{experiment}'")
        print("Available experiments:")
        for config in config_manager.get_all_experiment_configs():
            print(f"  - {config.name}: {config.description}")
        return
    
    if len(suites) == 0:
        suites = ("banking",)  # Default to banking suite
    
    if len(suites) != 1:
        print(f"Benchmarking suites {', '.join(suites)} with experiment: {experiment}")
    
    if len(suites) != 1 and len(user_tasks) != 0:
        raise ValueError("A user task can be specified only when one suite is being executed")
    
    # Create results directory
    logdir.mkdir(exist_ok=True)
    
    if max_workers == 1:
        all_results = {}
        all_metrics = {}
        
        for suite_name in suites:
            suite = get_suite(benchmark_version, suite_name)
            results, metrics = benchmark_suite_multi_model(
                experiment_name=experiment,
                suite=suite,
                logdir=logdir,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                attack=attack,
                defense=defense,
                system_message_name=system_message_name,
                system_message=system_message,
                force_rerun=force_rerun,
            )
            all_results[suite_name] = results
            all_metrics[suite_name] = metrics
        
        # Show results
        for suite_name, results in all_results.items():
            metrics = all_metrics[suite_name]
            show_multi_model_results(experiment, suite_name, results, metrics, attack is not None)
        
        # Save results if requested
        if save_results:
            results_file = logdir / f"{experiment}_results.json"
            combined_results = {
                "experiment": experiment,
                "results": all_results,
                "metrics": all_metrics,
                "timestamp": time.time(),
            }
            
            with open(results_file, 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)
            
            print(f"\nResults saved to: {results_file}")
        
        return
    
    # Multi-processing not implemented for multi-model experiments yet
    print("Multi-processing not yet supported for multi-model experiments")
    return


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()