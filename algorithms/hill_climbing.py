from typing import Dict, Any, Tuple
import time
from tqdm import tqdm
from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function
from lightup.utils import random_choice_with_weights


def solve(board: Board, max_iterations: int = 1000, max_time: float = 60.0, 
          stochastic: bool = False, neighbor_count: int = 10, 
          restart_count: int = 5, verbose: bool = False, **kwargs) -> Tuple[Solution, Dict[str, Any]]:
   
    start_time = time.time()
    
    best_solution_overall = Solution.random_solution(board)
    best_score_overall, _ = objective_function(best_solution_overall)
    
    stats = {
        "algorithm": "hill_climbing",
        "variant": "stochastic" if stochastic else "deterministic",
        "iterations": 0,
        "restarts": 0,
        "plateau_iterations": 0,
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
        
        if verbose:
            print(f"Restart {restart + 1}/{restart_count}, initial score: {current_score}")
        
        stats["restart_scores"].append(current_score)
        
        plateau_iterations = 0
        
        for iteration in tqdm(range(max_iterations), disable=not verbose):
            stats["iterations"] += 1
            
            if time.time() - start_time > max_time:
                if verbose:
                    print(f"Time limit of {max_time} seconds reached.")
                break
            
            neighbors = current_solution.get_smart_neighbors(count=neighbor_count)
            stats["total_neighbors_generated"] += len(neighbors)
            
            neighbor_scores = []
            for neighbor in neighbors:
                score, _ = objective_function(neighbor)
                neighbor_scores.append((neighbor, score))
            
            if stochastic:
                max_score = max(score for _, score in neighbor_scores) + 1
                weights = [max_score - score for _, score in neighbor_scores]
                
                chosen_index = random_choice_with_weights(
                    list(range(len(neighbor_scores))), 
                    weights
                )
                next_solution, next_score = neighbor_scores[chosen_index]
            else:
                next_solution, next_score = min(neighbor_scores, key=lambda x: x[1])
            
            stats["convergence_curve"].append(next_score)
            
            if next_score < current_score:
                current_solution = next_solution
                current_score = next_score
                plateau_iterations = 0
            else:
                plateau_iterations += 1
                stats["plateau_iterations"] += 1
                
                if plateau_iterations >= max_iterations // 10:
                    if verbose:
                        print(f"No improvement for {plateau_iterations} iterations. Breaking.")
                    break
            
            if current_score < best_score_overall:
                best_solution_overall = current_solution.copy()
                best_score_overall = current_score
                
                if verbose:
                    print(f"New best solution with score {best_score_overall}")
                
                if best_score_overall == 0:
                    if verbose:
                        print("Perfect solution found!")
                    break
        
        if best_score_overall == 0:
            break
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score_overall
    
    if verbose:
        print(f"Hill climbing completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['iterations']} iterations over {stats['restarts']} restarts")
        print(f"Generated {stats['total_neighbors_generated']} neighbors")
        print(f"Best solution has score {best_score_overall}")
    
    return best_solution_overall, stats 