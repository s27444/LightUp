from typing import Dict, Any, Tuple
import time
from collections import deque
from tqdm import tqdm
from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function, solution_difference


def solve(board: Board, max_iterations: int = 1000, max_time: float = 60.0,
          tabu_size: int = 10, neighbor_count: int = 20, aspiration_enabled: bool = True,
          backtrack_enabled: bool = False, restart_count: int = 3, 
          verbose: bool = False, **kwargs) -> Tuple[Solution, Dict[str, Any]]:
    start_time = time.time()
    
    best_solution_overall = Solution.random_solution(board)
    best_score_overall, _ = objective_function(best_solution_overall)
    
    stats = {
        "algorithm": "tabu_search",
        "iterations": 0,
        "restarts": 0,
        "aspiration_activations": 0,
        "backtrack_activations": 0,
        "total_neighbors_generated": 0,
        "execution_time": 0.0,
        "convergence_curve": [],
        "restart_scores": []
    }
    
    for restart in range(restart_count):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Time limit of {max_time} seconds reached.")
            break
        
        stats["restarts"] += 1
        
        current_solution = Solution.random_solution(board)
        current_score, _ = objective_function(current_solution)
        
        best_solution_restart = current_solution.copy()
        best_score_restart = current_score
        
        stats["restart_scores"].append(current_score)
        
        if verbose:
            print(f"Restart {restart + 1}/{restart_count}, initial score: {current_score}")
        
        tabu_list = deque(maxlen=tabu_size)
        
        previous_solutions = []
        
        for iteration in tqdm(range(max_iterations), disable=not verbose):
            stats["iterations"] += 1
            
            if time.time() - start_time > max_time:
                if verbose:
                    print(f"Time limit of {max_time} seconds reached.")
                break
            
            neighbors = current_solution.get_neighbors(count=neighbor_count)
            stats["total_neighbors_generated"] += len(neighbors)
            
            neighbor_scores = []
            for neighbor in neighbors:
                is_tabu = any(solution_difference(neighbor, tabu_solution) == 0 
                              for tabu_solution in tabu_list)
                
                score, _ = objective_function(neighbor)
                
                if not is_tabu or (aspiration_enabled and score < best_score_restart):
                    neighbor_scores.append((neighbor, score, is_tabu))
                    
                    if is_tabu and aspiration_enabled and score < best_score_restart:
                        stats["aspiration_activations"] += 1
            
            if not neighbor_scores:
                if backtrack_enabled and previous_solutions:
                    current_solution = previous_solutions[-1].copy()
                    current_score, _ = objective_function(current_solution)
                    previous_solutions.pop()
                    stats["backtrack_activations"] += 1
                    
                    if verbose:
                        print(f"All neighbors are tabu. Backtracking to solution with score {current_score}")
                    
                    continue
                else:
                    if verbose:
                        print("All neighbors are tabu. Skipping iteration.")
                    continue
            
            next_solution, next_score, is_tabu = min(neighbor_scores, key=lambda x: x[1])
            
            stats["convergence_curve"].append(next_score)
            
            if backtrack_enabled:
                previous_solutions.append(current_solution.copy())
                if len(previous_solutions) > tabu_size * 2:
                    previous_solutions.pop(0)
            
            current_solution = next_solution
            current_score = next_score
            
            tabu_list.append(current_solution.copy())
            
            if current_score < best_score_restart:
                best_solution_restart = current_solution.copy()
                best_score_restart = current_score
                
                if verbose:
                    print(f"New best solution for this restart with score {best_score_restart}")
                
                if best_score_restart == 0:
                    if verbose:
                        print("Perfect solution found!")
                    break
            
            if best_score_restart < best_score_overall:
                best_solution_overall = best_solution_restart.copy()
                best_score_overall = best_score_restart
                
                if verbose:
                    print(f"New best overall solution with score {best_score_overall}")
        
        if best_score_overall == 0:
            break
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score_overall
    
    if verbose:
        print(f"Tabu search completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['iterations']} iterations over {stats['restarts']} restarts")
        print(f"Generated {stats['total_neighbors_generated']} neighbors")
        print(f"Aspiration activated {stats['aspiration_activations']} times")
        if backtrack_enabled:
            print(f"Backtracking activated {stats['backtrack_activations']} times")
        print(f"Best solution has score {best_score_overall}")
    
    return best_solution_overall, stats