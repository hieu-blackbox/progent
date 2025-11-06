#!/usr/bin/env python3
"""
Multi-Model Experiment Runner for AgentDojo Banking Suite

This script runs comprehensive experiments comparing different model combinations
for policy generation vs agent execution in the AgentDojo banking suite.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import subprocess
from datetime import datetime

from multi_model_config import get_config_manager, ExperimentConfig
from multi_model_benchmark import main as benchmark_main


class ExperimentRunner:
    """Orchestrates multi-model experiments and result analysis."""
    
    def __init__(self, output_dir: Path = Path("experiment_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.config_manager = get_config_manager()
        self.results = {}
        
    def run_single_experiment(
        self,
        experiment_name: str,
        attack: str = None,
        defense: str = None,
        user_tasks: List[str] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Run a single multi-model experiment."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Attack: {attack or 'None'}")
        print(f"Defense: {defense or 'None'}")
        print(f"{'='*60}")
        
        # Prepare command arguments
        cmd = [
            sys.executable, "multi_model_benchmark.py",
            "--experiment", experiment_name,
            "--suite", "banking",
            "--save-results"
        ]
        
        if attack:
            cmd.extend(["--attack", attack])
        if defense:
            cmd.extend(["--defense", defense])
        if force_rerun:
            cmd.append("--force-rerun")
        if user_tasks:
            for task in user_tasks:
                cmd.extend(["--user-task", task])
        
        # Set log directory
        log_dir = self.output_dir / "logs" / experiment_name
        cmd.extend(["--logdir", str(log_dir)])
        
        try:
            # Run the experiment
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=3600  # 1 hour timeout
            )
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"Error running experiment {experiment_name}:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "experiment_name": experiment_name,
                    "status": "failed",
                    "error": result.stderr,
                    "duration": end_time - start_time
                }
            
            # Try to load the saved results
            results_file = self.output_dir / "logs" / experiment_name / f"{experiment_name}_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    experiment_results = json.load(f)
            else:
                experiment_results = {"raw_output": result.stdout}
            
            experiment_results.update({
                "experiment_name": experiment_name,
                "status": "completed",
                "duration": end_time - start_time,
                "attack": attack,
                "defense": defense,
                "stdout": result.stdout,
            })
            
            return experiment_results
            
        except subprocess.TimeoutExpired:
            return {
                "experiment_name": experiment_name,
                "status": "timeout",
                "duration": 3600,
                "attack": attack,
                "defense": defense
            }
        except Exception as e:
            return {
                "experiment_name": experiment_name,
                "status": "error",
                "error": str(e),
                "attack": attack,
                "defense": defense
            }
    
    def run_experiment_suite(
        self,
        experiments: List[str] = None,
        attacks: List[str] = None,
        defenses: List[str] = None,
        user_tasks: List[str] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Run a suite of experiments with different configurations."""
        
        if experiments is None:
            # Use a subset of experiments for the demo
            experiments = [
                "baseline_gpt4o",
                "cost_optimized_gpt", 
                "mixed_providers",
                "ultra_cheap_policy"
            ]
        
        if attacks is None:
            attacks = [None, "important_instructions"]  # No attack and with attack
        
        if defenses is None:
            defenses = [None]  # No defense for now
        
        all_results = {}
        total_experiments = len(experiments) * len(attacks) * len(defenses)
        current_experiment = 0
        
        print(f"Starting experiment suite with {total_experiments} total experiments")
        print(f"Experiments: {experiments}")
        print(f"Attacks: {attacks}")
        print(f"Defenses: {defenses}")
        
        suite_start_time = time.time()
        
        for experiment_name in experiments:
            for attack in attacks:
                for defense in defenses:
                    current_experiment += 1
                    print(f"\nProgress: {current_experiment}/{total_experiments}")
                    
                    # Create unique key for this configuration
                    config_key = f"{experiment_name}_{attack or 'no_attack'}_{defense or 'no_defense'}"
                    
                    # Run the experiment
                    result = self.run_single_experiment(
                        experiment_name=experiment_name,
                        attack=attack,
                        defense=defense,
                        user_tasks=user_tasks,
                        force_rerun=force_rerun
                    )
                    
                    all_results[config_key] = result
                    
                    # Save intermediate results
                    self.save_results(all_results, "intermediate_results.json")
        
        suite_end_time = time.time()
        
        # Add suite metadata
        suite_results = {
            "suite_metadata": {
                "total_experiments": total_experiments,
                "total_duration": suite_end_time - suite_start_time,
                "timestamp": datetime.now().isoformat(),
                "experiments": experiments,
                "attacks": attacks,
                "defenses": defenses,
            },
            "results": all_results
        }
        
        return suite_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save experiment results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_model_experiment_results_{timestamp}.json"
        
        results_file = self.output_dir / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        return results_file
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and generate summary statistics."""
        analysis = {
            "summary": {},
            "comparisons": {},
            "cost_analysis": {},
            "performance_analysis": {}
        }
        
        # Extract results by experiment type
        experiment_results = {}
        for config_key, result in results.get("results", {}).items():
            if result.get("status") != "completed":
                continue
                
            experiment_name = result.get("experiment_name")
            if experiment_name not in experiment_results:
                experiment_results[experiment_name] = []
            experiment_results[experiment_name].append(result)
        
        # Calculate summary statistics
        for experiment_name, exp_results in experiment_results.items():
            if not exp_results:
                continue
                
            # Calculate average utility and security
            utilities = []
            securities = []
            durations = []
            
            for result in exp_results:
                duration = result.get("duration", 0)
                durations.append(duration)
                
                # Try to extract utility/security from results
                # This would need to be adapted based on actual result structure
                if "results" in result:
                    suite_results = result["results"]
                    if "banking" in suite_results:
                        banking_results = suite_results["banking"]
                        if "utility_results" in banking_results:
                            utility_values = list(banking_results["utility_results"].values())
                            if utility_values:
                                utilities.append(sum(utility_values) / len(utility_values))
                        if "security_results" in banking_results:
                            security_values = list(banking_results["security_results"].values())
                            if security_values:
                                securities.append(sum(security_values) / len(security_values))
            
            analysis["summary"][experiment_name] = {
                "num_runs": len(exp_results),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "avg_utility": sum(utilities) / len(utilities) if utilities else 0,
                "avg_security": sum(securities) / len(securities) if securities else 0,
            }
        
        return analysis
    
    def generate_report(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate a human-readable report of the experiment results."""
        report = []
        report.append("Multi-Model Experiment Report")
        report.append("=" * 50)
        report.append("")
        
        # Suite metadata
        if "suite_metadata" in results:
            metadata = results["suite_metadata"]
            report.append(f"Total Experiments: {metadata.get('total_experiments', 'Unknown')}")
            report.append(f"Total Duration: {metadata.get('total_duration', 0):.2f} seconds")
            report.append(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
            report.append("")
        
        # Summary statistics
        if "summary" in analysis:
            report.append("Experiment Summary:")
            report.append("-" * 30)
            for experiment_name, stats in analysis["summary"].items():
                report.append(f"\n{experiment_name}:")
                report.append(f"  Runs: {stats['num_runs']}")
                report.append(f"  Avg Duration: {stats['avg_duration']:.2f}s")
                report.append(f"  Avg Utility: {stats['avg_utility']:.2%}")
                report.append(f"  Avg Security: {stats['avg_security']:.2%}")
        
        # Failed experiments
        failed_experiments = []
        for config_key, result in results.get("results", {}).items():
            if result.get("status") != "completed":
                failed_experiments.append((config_key, result.get("status"), result.get("error", "")))
        
        if failed_experiments:
            report.append("\nFailed Experiments:")
            report.append("-" * 30)
            for config_key, status, error in failed_experiments:
                report.append(f"{config_key}: {status}")
                if error:
                    report.append(f"  Error: {error[:100]}...")
        
        return "\n".join(report)


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run multi-model experiments on AgentDojo banking suite")
    parser.add_argument(
        "--experiments", 
        nargs="+", 
        help="List of experiment names to run (default: subset of available experiments)"
    )
    parser.add_argument(
        "--attacks", 
        nargs="+", 
        default=[None, "important_instructions"],
        help="List of attacks to test (default: None and important_instructions)"
    )
    parser.add_argument(
        "--defenses", 
        nargs="+", 
        default=[None],
        help="List of defenses to test (default: None)"
    )
    parser.add_argument(
        "--user-tasks", 
        nargs="+", 
        help="Specific user tasks to run (default: all tasks)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("experiment_results"),
        help="Directory to save results (default: experiment_results)"
    )
    parser.add_argument(
        "--force-rerun", 
        action="store_true",
        help="Force rerun of experiments even if results exist"
    )
    parser.add_argument(
        "--list-experiments", 
        action="store_true",
        help="List available experiments and exit"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run a quick test with minimal experiments"
    )
    
    args = parser.parse_args()
    
    if args.list_experiments:
        config_manager = get_config_manager()
        config_manager.print_experiment_summary()
        return
    
    # Create experiment runner
    runner = ExperimentRunner(args.output_dir)
    
    # Configure experiments for quick test
    if args.quick_test:
        experiments = ["baseline_gpt4o", "cost_optimized_gpt"]
        attacks = [None]  # No attacks for quick test
        defenses = [None]
        user_tasks = ["user_task_0", "user_task_1"]  # Just a couple tasks
    else:
        experiments = args.experiments
        attacks = args.attacks
        defenses = args.defenses
        user_tasks = args.user_tasks
    
    print("Starting multi-model experiment suite...")
    print(f"Output directory: {args.output_dir}")
    
    # Run the experiment suite
    results = runner.run_experiment_suite(
        experiments=experiments,
        attacks=attacks,
        defenses=defenses,
        user_tasks=user_tasks,
        force_rerun=args.force_rerun
    )
    
    # Save final results
    results_file = runner.save_results(results)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = runner.analyze_results(results)
    
    # Generate and save report
    report = runner.generate_report(results, analysis)
    report_file = args.output_dir / "experiment_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nExperiment Report:")
    print(report)
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()