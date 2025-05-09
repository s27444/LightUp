import os
import sys
import time
import argparse
import json
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function
from lightup.utils import format_time, format_solution_stats
import algorithms.brute_force as brute_force
import algorithms.hill_climbing as hill_climbing
import algorithms.tabu_search as tabu_search
import algorithms.simulated_annealing as simulated_annealing
import algorithms.genetic_algorithm as genetic_algorithm
import algorithms.parallel_genetic as parallel_genetic
import algorithms.island_genetic as island_genetic


def run_experiment(board: Board, algorithms: Dict[str, Dict[str, Any]], 
                  output_dir: str = None, verbose: bool = False) -> Dict[str, Any]:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "board_size": (board.width, board.height),
        "white_cells_count": len(board.get_white_cells()),
        "black_cells_count": sum(1 for y in range(board.height) for x in range(board.width) if board.is_black_cell(x, y)),
        "algorithms": {}
    }
    
    for algo_name, params in algorithms.items():
        if verbose:
            print(f"\n{'=' * 80}\nRunning {algo_name} with parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        if algo_name == "brute_force":
            solve_func = brute_force.solve
        elif algo_name == "hill_climbing":
            solve_func = hill_climbing.solve
        elif algo_name == "tabu_search":
            solve_func = tabu_search.solve
        elif algo_name == "simulated_annealing":
            solve_func = simulated_annealing.solve
        elif algo_name == "genetic_algorithm":
            solve_func = genetic_algorithm.solve
        elif algo_name == "parallel_genetic":
            solve_func = parallel_genetic.solve
        elif algo_name == "island_genetic":
            solve_func = island_genetic.solve
        else:
            if verbose:
                print(f"Unknown algorithm: {algo_name}")
            continue
        
        try:
            start_time = time.time()
            solution, stats = solve_func(board=board, verbose=verbose, **params)
            execution_time = time.time() - start_time
            
            score, metrics = objective_function(solution)
            
            algo_results = {
                "execution_time": execution_time,
                "solution_score": score,
                "solution_metrics": metrics,
                "stats": stats
            }
            
            results["algorithms"][algo_name] = algo_results
            
            if verbose:
                print(f"\nResults for {algo_name}:")
                print(f"Execution time: {format_time(execution_time)}")
                print(format_solution_stats(metrics))
                print(f"Solution:")
                print(solution)
        
        except Exception as e:
            if verbose:
                print(f"Error running {algo_name}: {e}")
    
    if output_dir:
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            serializable_results = {}
            for key, value in results.items():
                if key == "algorithms":
                    serializable_results[key] = {}
                    for algo_name, algo_results in value.items():
                        serializable_results[key][algo_name] = {}
                        for result_key, result_value in algo_results.items():
                            if result_key == "stats":
                                serializable_stats = {}
                                for stat_key, stat_value in result_value.items():
                                    if isinstance(stat_value, np.ndarray):
                                        serializable_stats[stat_key] = stat_value.tolist()
                                    elif isinstance(stat_value, (int, float, str, bool, list, dict)) or stat_value is None:
                                        serializable_stats[stat_key] = stat_value
                                    else:
                                        serializable_stats[stat_key] = str(stat_value)
                                serializable_results[key][algo_name][result_key] = serializable_stats
                            else:
                                serializable_results[key][algo_name][result_key] = result_value
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print(f"Results saved to {results_file}")
    
    return results


def plot_results(results: Dict[str, Any], output_dir: str = None, 
                 show_plots: bool = True) -> None:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    algorithm_names = list(results["algorithms"].keys())
    if not algorithm_names:
        print("No algorithm results to plot")
        return
    
    plt.figure(figsize=(10, 6))
    execution_times = [results["algorithms"][algo]["execution_time"] for algo in algorithm_names]
    plt.bar(algorithm_names, execution_times)
    plt.title("Execution Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "execution_time.png"))
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    plt.figure(figsize=(10, 6))
    solution_scores = [results["algorithms"][algo]["solution_score"] for algo in algorithm_names]
    plt.bar(algorithm_names, solution_scores)
    plt.title("Solution Quality Comparison (lower is better)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "solution_quality.png"))
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    convergence_data = {}
    for algo in algorithm_names:
        if "stats" in results["algorithms"][algo]:
            stats = results["algorithms"][algo]["stats"]
            if "convergence_curve" in stats and stats["convergence_curve"]:
                convergence_data[algo] = stats["convergence_curve"]
            elif "best_fitness_history" in stats and stats["best_fitness_history"]:
                convergence_data[algo] = stats["best_fitness_history"]
    
    if convergence_data:
        plt.figure(figsize=(12, 8))
        
        max_iterations = max(len(curve) for curve in convergence_data.values())
        for algo, curve in convergence_data.items():
            if len(curve) > 100:
                indices = np.linspace(0, len(curve)-1, 100).astype(int)
                convergence_data[algo] = [curve[i] for i in indices]
        
        for algo, curve in convergence_data.items():
            plt.plot(curve, label=algo)
        
        plt.title("Convergence Curves")
        plt.xlabel("Iteration")
        plt.ylabel("Score (lower is better)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "convergence_curves.png"))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(solution_scores, execution_times, s=100, alpha=0.7)
    
    for i, algo in enumerate(algorithm_names):
        plt.annotate(algo, (solution_scores[i], execution_times[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title("Resource Usage vs. Solution Quality")
    plt.xlabel("Solution Score (lower is better)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "resource_vs_quality.png"))
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    summary_data = []
    for algo in algorithm_names:
        algo_results = results["algorithms"][algo]
        summary_data.append({
            "Algorithm": algo,
            "Execution Time (s)": algo_results["execution_time"],
            "Solution Score": algo_results["solution_score"],
            "Valid Solution": algo_results["solution_metrics"]["is_valid"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_df = summary_df.sort_values(["Solution Score", "Execution Time (s)"])
    
    print("\nAlgorithm Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Compare optimization methods for Light Up puzzles")
    
    parser.add_argument("--input", "-i", required=True, help="Input board file")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    
    parser.add_argument("--algorithms", "-a", nargs="+", 
                        default=["hill_climbing", "tabu_search", "simulated_annealing", "genetic_algorithm"],
                        choices=["brute_force", "hill_climbing", "tabu_search", 
                                "simulated_annealing", "genetic_algorithm", 
                                "parallel_genetic", "island_genetic"],
                        help="Algorithms to compare")
    
    parser.add_argument("--time-limit", "-t", type=float, default=60.0, 
                       help="Time limit per algorithm in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-plots", action="store_true", help="Don't show plots")
    
    parser.add_argument("--hc-neighbor-count", type=int, default=10, 
                       help="Number of neighbors for hill climbing")
    parser.add_argument("--hc-restart-count", type=int, default=5, 
                       help="Number of restarts for hill climbing")
    
    parser.add_argument("--ts-tabu-size", type=int, default=10, 
                       help="Tabu list size for tabu search")
    parser.add_argument("--ts-neighbor-count", type=int, default=20, 
                       help="Number of neighbors for tabu search")
    parser.add_argument("--ts-restart-count", type=int, default=3, 
                       help="Number of restarts for tabu search")
    
    parser.add_argument("--sa-initial-temp", type=float, default=1000.0, 
                       help="Initial temperature for simulated annealing")
    parser.add_argument("--sa-cooling-rate", type=float, default=0.001, 
                       help="Cooling rate for simulated annealing")
    parser.add_argument("--sa-restart-count", type=int, default=3, 
                       help="Number of restarts for simulated annealing")
    
    parser.add_argument("--ga-population-size", type=int, default=50, 
                       help="Population size for genetic algorithm")
    parser.add_argument("--ga-max-generations", type=int, default=100, 
                       help="Maximum number of generations for genetic algorithm")
    parser.add_argument("--ga-elite-size", type=int, default=5, 
                       help="Elite size for genetic algorithm")
    
    parser.add_argument("--pg-population-size", type=int, default=50, 
                       help="Population size for parallel genetic algorithm")
    parser.add_argument("--pg-max-generations", type=int, default=100, 
                       help="Maximum number of generations for parallel genetic algorithm")
    parser.add_argument("--pg-elite-size", type=int, default=5, 
                       help="Elite size for parallel genetic algorithm")
    parser.add_argument("--pg-num-workers", type=int, default=None, 
                       help="Number of worker processes for parallel genetic algorithm")
    
    parser.add_argument("--ig-population-size", type=int, default=50, 
                       help="Population size for island genetic algorithm")
    parser.add_argument("--ig-max-generations", type=int, default=100, 
                       help="Maximum number of generations for island genetic algorithm")
    parser.add_argument("--ig-elite-size", type=int, default=5, 
                       help="Elite size for island genetic algorithm")
    parser.add_argument("--ig-num-islands", type=int, default=4, 
                       help="Number of islands for island genetic algorithm")
    parser.add_argument("--ig-migration-interval", type=int, default=10, 
                       help="Migration interval for island genetic algorithm")
    parser.add_argument("--ig-migration-rate", type=float, default=0.1, 
                       help="Migration rate for island genetic algorithm")
    
    args = parser.parse_args()
    
    try:
        board = Board.from_file(args.input)
    except Exception as e:
        print(f"Error loading board: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loaded board of size {board.width}x{board.height}")
        print(f"White cells: {len(board.get_white_cells())}")
        print(f"Algorithms to compare: {', '.join(args.algorithms)}")
    
    algorithms = {}
    
    if "brute_force" in args.algorithms:
        algorithms["brute_force"] = {
            "max_time": args.time_limit,
            "max_solutions": 10000
        }
    
    if "hill_climbing" in args.algorithms:
        algorithms["hill_climbing"] = {
            "max_time": args.time_limit,
            "stochastic": True,
            "neighbor_count": args.hc_neighbor_count,
            "restart_count": args.hc_restart_count
        }
    
    if "tabu_search" in args.algorithms:
        algorithms["tabu_search"] = {
            "max_time": args.time_limit,
            "tabu_size": args.ts_tabu_size,
            "aspiration_enabled": True,
            "backtrack_enabled": True,
            "neighbor_count": args.ts_neighbor_count,
            "restart_count": args.ts_restart_count
        }
    
    if "simulated_annealing" in args.algorithms:
        algorithms["simulated_annealing"] = {
            "max_time": args.time_limit,
            "initial_temp": args.sa_initial_temp,
            "cooling_rate": args.sa_cooling_rate,
            "cooling_schedule": "exponential",
            "restart_count": args.sa_restart_count
        }
    
    if "genetic_algorithm" in args.algorithms:
        algorithms["genetic_algorithm"] = {
            "max_time": args.time_limit,
            "population_size": args.ga_population_size,
            "max_generations": args.ga_max_generations,
            "crossover_method": "uniform",
            "mutation_method": "random_flip",
            "elite_size": args.ga_elite_size
        }
    
    if "parallel_genetic" in args.algorithms:
        algorithms["parallel_genetic"] = {
            "max_time": args.time_limit,
            "population_size": args.pg_population_size,
            "max_generations": args.pg_max_generations,
            "crossover_method": "uniform",
            "mutation_method": "random_flip",
            "elite_size": args.pg_elite_size,
            "num_workers": args.pg_num_workers
        }
    
    if "island_genetic" in args.algorithms:
        algorithms["island_genetic"] = {
            "max_time": args.time_limit,
            "population_size": args.ig_population_size,
            "max_generations": args.ig_max_generations,
            "crossover_method": "uniform",
            "mutation_method": "random_flip",
            "elite_size": args.ig_elite_size,
            "num_islands": args.ig_num_islands,
            "migration_interval": args.ig_migration_interval,
            "migration_rate": args.ig_migration_rate
        }
    
    results = run_experiment(board, algorithms, args.output, args.verbose)
    
    if not args.no_plots:
        plot_results(results, args.output, show_plots=True)
    else:
        plot_results(results, args.output, show_plots=False)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 