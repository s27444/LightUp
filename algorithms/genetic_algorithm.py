from typing import Dict, Any, List, Tuple, Set, Callable, Optional
import time
import random
import copy
from tqdm import tqdm
import numpy as np
from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function


def uniform_crossover(parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
    child1 = Solution(parent1.board)
    child2 = Solution(parent1.board)
    
    white_cells = parent1.board.get_white_cells()
    
    for x, y in white_cells:
        if random.random() < 0.5:
            if (x, y) in parent1.bulbs:
                child1.place_bulb(x, y)
            if (x, y) in parent2.bulbs:
                child2.place_bulb(x, y)
        else:
            if (x, y) in parent2.bulbs:
                child1.place_bulb(x, y)
            if (x, y) in parent1.bulbs:
                child2.place_bulb(x, y)
    
    return child1, child2


def single_point_crossover(parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
    child1 = Solution(parent1.board)
    child2 = Solution(parent1.board)
    
    white_cells = sorted(parent1.board.get_white_cells())
    
    if len(white_cells) <= 1:
        return parent1.copy(), parent2.copy()
    
    crossover_point = random.randint(1, len(white_cells) - 1)
    
    for i, (x, y) in enumerate(white_cells):
        if i < crossover_point:
            if (x, y) in parent1.bulbs:
                child1.place_bulb(x, y)
            if (x, y) in parent2.bulbs:
                child2.place_bulb(x, y)
        else:
            if (x, y) in parent2.bulbs:
                child1.place_bulb(x, y)
            if (x, y) in parent1.bulbs:
                child2.place_bulb(x, y)
    
    return child1, child2


def random_flip_mutation(solution: Solution, mutation_rate: float = 0.1) -> Solution:
    mutated = solution.copy()
    
    white_cells = solution.board.get_white_cells()
    
    for x, y in white_cells:
        if random.random() < mutation_rate:
            if (x, y) in mutated.bulbs:
                mutated.remove_bulb(x, y)
            else:
                mutated.place_bulb(x, y)
    
    return mutated


def swap_mutation(solution: Solution, mutation_rate: float = 0.1) -> Solution:
    mutated = solution.copy()
    
    white_cells = solution.board.get_white_cells()
    
    num_swaps = int(len(white_cells) * mutation_rate / 2)
    
    for _ in range(num_swaps):
        if len(white_cells) < 2:
            break
        
        pos1, pos2 = random.sample(white_cells, 2)
        
        has_bulb1 = pos1 in mutated.bulbs
        has_bulb2 = pos2 in mutated.bulbs
        
        if has_bulb1:
            mutated.remove_bulb(*pos1)
        else:
            mutated.place_bulb(*pos1)
            
        if has_bulb2:
            mutated.remove_bulb(*pos2)
        else:
            mutated.place_bulb(*pos2)
    
    return mutated


def solve(board: Board, population_size: int = 50, max_generations: int = 100,
          crossover_method: str = "uniform", mutation_method: str = "random_flip",
          crossover_rate: float = 0.8, mutation_rate: float = 0.1,
          elite_size: int = 5, tournament_size: int = 3, 
          termination_condition: str = "iterations", convergence_generations: int = 20,
          max_time: float = 300.0, verbose: bool = False, **kwargs) -> Tuple[Solution, Dict[str, Any]]:

    start_time = time.time()
    
    if crossover_method == "uniform":
        crossover_func = uniform_crossover
    elif crossover_method == "single_point":
        crossover_func = single_point_crossover
    else:
        raise ValueError(f"Unknown crossover method: {crossover_method}")
    
    if mutation_method == "random_flip":
        mutation_func = lambda sol: random_flip_mutation(sol, mutation_rate)
    elif mutation_method == "swap":
        mutation_func = lambda sol: swap_mutation(sol, mutation_rate)
    else:
        raise ValueError(f"Unknown mutation method: {mutation_method}")
    
    population = [Solution.random_solution(board) for _ in range(population_size)]
    
    population_scores = []
    for solution in population:
        score, _ = objective_function(solution)
        population_scores.append((solution, score))
    
    population_scores.sort(key=lambda x: x[1])
    
    best_solution = population_scores[0][0].copy()
    best_score = population_scores[0][1]
    
    stats = {
        "algorithm": "genetic_algorithm",
        "generations": 0,
        "generations_without_improvement": 0,
        "best_fitness_history": [best_score],
        "avg_fitness_history": [sum(score for _, score in population_scores) / len(population_scores)],
        "execution_time": 0.0,
        "crossover_method": crossover_method,
        "mutation_method": mutation_method,
        "population_size": population_size,
        "elite_size": elite_size
    }
    
    if verbose:
        print(f"Initial population created with {population_size} solutions")
        print(f"Best initial solution has score {best_score}")
        print(f"Starting evolution...")
    
    for generation in tqdm(range(max_generations), disable=not verbose):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Time limit of {max_time} seconds reached.")
            break
        
        stats["generations"] += 1
        
        next_generation = []
        
        for i in range(min(elite_size, len(population_scores))):
            next_generation.append(population_scores[i][0].copy())
        
        while len(next_generation) < population_size:
            parent1 = tournament_selection(population_scores, tournament_size)
            parent2 = tournament_selection(population_scores, tournament_size)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover_func(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = mutation_func(child1)
            child2 = mutation_func(child2)
            
            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)
        
        population = next_generation
        
        population_scores = []
        for solution in population:
            score, _ = objective_function(solution)
            population_scores.append((solution, score))
        
        population_scores.sort(key=lambda x: x[1])
        
        current_best_score = population_scores[0][1]
        if current_best_score < best_score:
            best_solution = population_scores[0][0].copy()
            best_score = current_best_score
            stats["generations_without_improvement"] = 0
            
            if verbose:
                print(f"Generation {generation + 1}: New best solution with score {best_score}")
                
            if best_score == 0:
                if verbose:
                    print("Perfect solution found!")
                break
        else:
            stats["generations_without_improvement"] += 1
        
        stats["best_fitness_history"].append(population_scores[0][1])
        stats["avg_fitness_history"].append(sum(score for _, score in population_scores) / len(population_scores))
        
        if termination_condition == "convergence" and stats["generations_without_improvement"] >= convergence_generations:
            if verbose:
                print(f"No improvement for {convergence_generations} generations. Stopping.")
            break
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score
    
    if verbose:
        print(f"Genetic algorithm completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['generations']} generations")
        print(f"Best solution has score {best_score}")
    
    return best_solution, stats


def tournament_selection(population_scores: List[Tuple[Solution, float]], tournament_size: int) -> Solution:
    tournament = random.sample(population_scores, min(tournament_size, len(population_scores)))
    
    return min(tournament, key=lambda x: x[1])[0] 