from typing import Dict, Any, List, Tuple, Callable, Optional
import time
import math
import random
from tqdm import tqdm
from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function


def linear_cooling(initial_temp: float, alpha: float) -> Callable[[int], float]:
    def temperature(k: int) -> float:
        return max(0.1, initial_temp * (1 - alpha * k))
    return temperature


def exponential_cooling(initial_temp: float, alpha: float) -> Callable[[int], float]:
    def temperature(k: int) -> float:
        return max(0.1, initial_temp * (alpha ** k))
    return temperature


def logarithmic_cooling(initial_temp: float, alpha: float) -> Callable[[int], float]:
    def temperature(k: int) -> float:
        return max(0.1, initial_temp / (1 + alpha * math.log(1 + k)))
    return temperature


def boltzmann_cooling(initial_temp: float, alpha: float) -> Callable[[int], float]:
    def temperature(k: int) -> float:
        return max(0.1, initial_temp / (alpha * math.log(1 + k + 1)))
    return temperature


def cauchy_cooling(initial_temp: float, alpha: float) -> Callable[[int], float]:
    def temperature(k: int) -> float:
        return max(0.1, initial_temp / (1 + alpha * k))
    return temperature


def solve(board: Board, max_iterations: int = 10000, max_time: float = 60.0,
          initial_temp: float = 100.0, cooling_rate: float = 0.01,
          cooling_schedule: str = "exponential", restart_count: int = 3,
          use_normal_distribution: bool = False, sigma: float = 0.2,
          verbose: bool = False, **kwargs) -> Tuple[Solution, Dict[str, Any]]:
    start_time = time.time()
    
    cooling_schedules = {
        "linear": linear_cooling,
        "exponential": exponential_cooling,
        "logarithmic": logarithmic_cooling,
        "boltzmann": boltzmann_cooling,
        "cauchy": cauchy_cooling
    }
    
    if cooling_schedule not in cooling_schedules:
        raise ValueError(f"Unknown cooling schedule: {cooling_schedule}. Available schedules: {list(cooling_schedules.keys())}")
    
    if cooling_schedule == "exponential":
        temp_func = cooling_schedules[cooling_schedule](initial_temp, 1 - cooling_rate)
    else:
        temp_func = cooling_schedules[cooling_schedule](initial_temp, cooling_rate)
    
    best_solution_overall = Solution.random_solution(board)
    best_score_overall, _ = objective_function(best_solution_overall)
    
    stats = {
        "algorithm": "simulated_annealing",
        "iterations": 0,
        "restarts": 0,
        "accepted_worse": 0,
        "rejected_worse": 0,
        "accepted_better": 0,
        "execution_time": 0.0,
        "cooling_schedule": cooling_schedule,
        "normal_distribution": use_normal_distribution,
        "sigma": sigma if use_normal_distribution else None,
        "convergence_curve": [],
        "restart_scores": [],
        "temperature_curve": [],
        "acceptance_ratio": []
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
        
        accepted_count = 0
        total_count = 0
        
        for iteration in tqdm(range(max_iterations), disable=not verbose):
            stats["iterations"] += 1
            
            if time.time() - start_time > max_time:
                if verbose:
                    print(f"Time limit of {max_time} seconds reached.")
                break
            
            temperature = temp_func(iteration)
            stats["temperature_curve"].append(temperature)
            
            if use_normal_distribution:
                neighbor = current_solution.get_neighbor_normal(sigma=sigma)
            else:
                neighbor = current_solution.get_neighbor()
            
            neighbor_score, _ = objective_function(neighbor)
            
            delta = neighbor_score - current_score
            total_count += 1
            
            accept = False
            if delta < 0:
                accept = True
                stats["accepted_better"] += 1
                accepted_count += 1
            else:
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    accept = True
                    stats["accepted_worse"] += 1
                    accepted_count += 1
                else:
                    stats["rejected_worse"] += 1
            
            stats["acceptance_ratio"].append(accepted_count / total_count if total_count > 0 else 0)
            stats["convergence_curve"].append(current_score)
            
            if accept:
                current_solution = neighbor
                current_score = neighbor_score
                
                if current_score < best_score_restart:
                    best_solution_restart = current_solution.copy()
                    best_score_restart = current_score
                    
                    if verbose and iteration % 100 == 0:
                        print(f"New best solution for this restart with score {best_score_restart}")
                    
                    if best_score_restart == 0:
                        if verbose:
                            print("Perfect solution found!")
                        break
            
            if temperature < 0.2:
                if verbose:
                    print(f"Temperature frozen at {temperature:.6f}")
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
        print(f"Simulated annealing completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['iterations']} iterations over {stats['restarts']} restarts")
        print(f"Accepted better: {stats['accepted_better']}, Accepted worse: {stats['accepted_worse']}, Rejected worse: {stats['rejected_worse']}")
        print(f"Using normal distribution: {use_normal_distribution}")
        if use_normal_distribution:
            print(f"Sigma value: {sigma}")
        print(f"Best solution has score {best_score_overall}")
    
    return best_solution_overall, stats