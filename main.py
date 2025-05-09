import argparse
import sys
import time
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(description="Light Up (Akari) puzzle solver")
    
    parser.add_argument("--input", "-i", required=True, help="Input board file")
    
    parser.add_argument("--algorithm", "-a", default="hill_climbing", 
                       choices=["brute_force", "hill_climbing", "tabu_search", 
                                "simulated_annealing", "genetic_algorithm", 
                                "parallel_genetic", "island_genetic"],
                       help="Optimization algorithm to use")
    
    parser.add_argument("--output", "-o", help="Output solution file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    parser.add_argument("--time-limit", "-t", type=float, default=60.0, 
                       help="Time limit in seconds")
    
    parser.add_argument("--max-solutions", type=int, default=1000000, 
                       help="Maximum number of solutions to check (brute force)")
    
    parser.add_argument("--stochastic", action="store_true", 
                       help="Use stochastic hill climbing")
    parser.add_argument("--neighbor-count", type=int, default=10, 
                       help="Number of neighbors to generate per iteration")
    parser.add_argument("--restart-count", type=int, default=5, 
                       help="Number of random restarts")
    
    parser.add_argument("--tabu-size", type=int, default=10, 
                       help="Size of the tabu list")
    parser.add_argument("--aspiration", action="store_true", 
                       help="Enable aspiration criteria in tabu search")
    parser.add_argument("--backtrack", action="store_true", 
                       help="Enable backtracking in tabu search")
    
    parser.add_argument("--initial-temp", type=float, default=1000.0, 
                       help="Initial temperature for simulated annealing")
    parser.add_argument("--cooling-rate", type=float, default=0.001, 
                       help="Cooling rate for simulated annealing")
    parser.add_argument("--cooling-schedule", 
                       choices=["linear", "exponential", "logarithmic"],
                       default="exponential", 
                       help="Cooling schedule for simulated annealing")
    
    parser.add_argument("--population-size", type=int, default=50, 
                       help="Population size for genetic algorithm")
    parser.add_argument("--max-generations", type=int, default=100, 
                       help="Maximum number of generations for genetic algorithm")
    parser.add_argument("--crossover-method", 
                       choices=["uniform", "single_point"],
                       default="uniform", 
                       help="Crossover method for genetic algorithm")
    parser.add_argument("--mutation-method", 
                       choices=["random_flip", "swap"],
                       default="random_flip", 
                       help="Mutation method for genetic algorithm")
    parser.add_argument("--elite-size", type=int, default=5, 
                       help="Number of elite solutions to preserve in genetic algorithm")
    
    parser.add_argument("--num-workers", type=int, default=None, 
                       help="Number of worker processes for parallel genetic algorithm")
    
    parser.add_argument("--num-islands", type=int, default=4, 
                       help="Number of islands for island genetic algorithm")
    parser.add_argument("--migration-interval", type=int, default=10, 
                       help="Number of generations between migrations")
    parser.add_argument("--migration-rate", type=float, default=0.1, 
                       help="Fraction of population to migrate")
    parser.add_argument("--distributed", action="store_true", 
                       help="Run islands on separate processes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        board = Board.from_file(args.input)
    except Exception as e:
        print(f"Error loading board: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loaded board of size {board.width}x{board.height}")
        print(f"White cells: {len(board.get_white_cells())}")
        print(f"Black cells with numbers: {len(board.get_black_cells())}")
        print(f"Solving using {args.algorithm}...")
        print(board)
        print()
    
    algorithm_params = {
        "verbose": args.verbose,
        "max_time": args.time_limit
    }
    
    if args.algorithm == "brute_force":
        algorithm_params["max_solutions"] = args.max_solutions
        solve_func = brute_force.solve
    
    elif args.algorithm == "hill_climbing":
        algorithm_params["stochastic"] = args.stochastic
        algorithm_params["neighbor_count"] = args.neighbor_count
        algorithm_params["restart_count"] = args.restart_count
        solve_func = hill_climbing.solve
    
    elif args.algorithm == "tabu_search":
        algorithm_params["tabu_size"] = args.tabu_size
        algorithm_params["aspiration_enabled"] = args.aspiration
        algorithm_params["backtrack_enabled"] = args.backtrack
        algorithm_params["neighbor_count"] = args.neighbor_count
        algorithm_params["restart_count"] = args.restart_count
        solve_func = tabu_search.solve
    
    elif args.algorithm == "simulated_annealing":
        algorithm_params["initial_temp"] = args.initial_temp
        algorithm_params["cooling_rate"] = args.cooling_rate
        algorithm_params["cooling_schedule"] = args.cooling_schedule
        algorithm_params["restart_count"] = args.restart_count
        solve_func = simulated_annealing.solve
    
    elif args.algorithm == "genetic_algorithm":
        algorithm_params["population_size"] = args.population_size
        algorithm_params["max_generations"] = args.max_generations
        algorithm_params["crossover_method"] = args.crossover_method
        algorithm_params["mutation_method"] = args.mutation_method
        algorithm_params["elite_size"] = args.elite_size
        solve_func = genetic_algorithm.solve
    
    elif args.algorithm == "parallel_genetic":
        algorithm_params["population_size"] = args.population_size
        algorithm_params["max_generations"] = args.max_generations
        algorithm_params["crossover_method"] = args.crossover_method
        algorithm_params["mutation_method"] = args.mutation_method
        algorithm_params["elite_size"] = args.elite_size
        algorithm_params["num_workers"] = args.num_workers
        solve_func = parallel_genetic.solve
    
    elif args.algorithm == "island_genetic":
        algorithm_params["population_size"] = args.population_size
        algorithm_params["max_generations"] = args.max_generations
        algorithm_params["crossover_method"] = args.crossover_method
        algorithm_params["mutation_method"] = args.mutation_method
        algorithm_params["elite_size"] = args.elite_size
        algorithm_params["num_islands"] = args.num_islands
        algorithm_params["migration_interval"] = args.migration_interval
        algorithm_params["migration_rate"] = args.migration_rate
        algorithm_params["distributed"] = args.distributed
        solve_func = island_genetic.solve
    
    else:
        print(f"Unknown algorithm: {args.algorithm}", file=sys.stderr)
        return 1
    
    start_time = time.time()
    solution, stats = solve_func(board, **algorithm_params)
    execution_time = time.time() - start_time
    
    score, metrics = objective_function(solution)
    
    print(f"\nRozwiązanie znalezione w czasie {format_time(execution_time)}")
    print(format_solution_stats(metrics))
    
    print("\nSolution:")
    print(solution)
    
    if args.verbose:
        print("\nStatistics:")
        print(f"Algorithm stats: {stats}")
    
    if args.output:
        try:
            output_path = Path(args.output)
            
            with open(output_path, "w") as f:
                f.write(f"{board.width} {board.height}\n")
                
                for y in range(board.height):
                    line = ""
                    for x in range(board.width):
                        if (x, y) in solution.bulbs:
                            line += "L"
                        else:
                            if board.grid[y][x] == 0:
                                line += "."
                            elif board.grid[y][x] == -1:
                                line += "#"
                            elif board.grid[y][x] == -2:
                                line += "0"
                            else:
                                line += str(board.grid[y][x])
                    f.write(line + "\n")
            
            if args.verbose:
                print(f"Solution saved to {args.output}")
        
        except Exception as e:
            print(f"Error saving solution: {e}", file=sys.stderr)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 