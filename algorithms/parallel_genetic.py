from typing import Dict, Any, Tuple
import time
import random
import copy
from tqdm import tqdm
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function
from algorithms.genetic_algorithm import (
    uniform_crossover, single_point_crossover,
    random_flip_mutation, swap_mutation, tournament_selection
)


def evaluate_individual(individual: Solution) -> Tuple[Solution, float, Dict[str, Any]]:
    score, metrics = objective_function(individual)
    return individual, score, metrics


def solve(board: Board, population_size: int = 50, max_generations: int = 100,
          crossover_method: str = "uniform", mutation_method: str = "random_flip",
          crossover_rate: float = 0.8, mutation_rate: float = 0.1,
          elite_size: int = 5, tournament_size: int = 3, 
          termination_condition: str = "iterations", convergence_generations: int = 20,
          max_time: float = 300.0, verbose: bool = False, 
          num_workers: int = None, **kwargs) -> Tuple[Solution, Dict[str, Any]]:
    start_time = time.time()
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    if verbose:
        print(f"Running parallel genetic algorithm with {num_workers} workers")
        print(f"Population size: {population_size}")
        print(f"Crossover method: {crossover_method}")
        print(f"Mutation method: {mutation_method}")
        print(f"Elite size: {elite_size}")
    
    if crossover_method == "uniform":
        crossover_func = uniform_crossover
    elif crossover_method == "single_point":
        crossover_func = single_point_crossover
    else:
        raise ValueError(f"Unknown crossover method: {crossover_method}")
    
    if mutation_method == "random_flip":
        mutation_func = random_flip_mutation
    elif mutation_method == "swap":
        mutation_func = swap_mutation
    else:
        raise ValueError(f"Unknown mutation method: {mutation_method}")
    
    population = [Solution.random_solution(board) for _ in range(population_size)]
    
    stats = {
        "algorithm": "parallel_genetic_algorithm",
        "generations": 0,
        "evaluations": 0,
        "best_score_history": [],
        "avg_score_history": [],
        "execution_time": 0.0,
        "workers": num_workers,
        "convergence_counter": 0
    }
    
    best_solution = None
    best_score = float("inf")
    
    generation_iterator = range(max_generations)
    if verbose:
        generation_iterator = tqdm(generation_iterator)
    
    for generation in generation_iterator:
        stats["generations"] += 1
        
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Time limit of {max_time} seconds reached.")
            break
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_individual = {
                executor.submit(evaluate_individual, individual): i 
                for i, individual in enumerate(population)
            }
            
            evaluated_population = [None] * len(population)
            scores = [None] * len(population)
            metrics_list = [None] * len(population)
            
            for future in as_completed(future_to_individual):
                i = future_to_individual[future]
                try:
                    individual, score, metrics = future.result()
                    evaluated_population[i] = individual
                    scores[i] = score
                    metrics_list[i] = metrics
                    stats["evaluations"] += 1
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating individual: {e}")
        
        population = evaluated_population
        
        population_scores = list(zip(population, scores))
        population_scores.sort(key=lambda x: x[1])
        
        current_best_solution, current_best_score = population_scores[0]
        
        if current_best_score < best_score:
            best_solution = current_best_solution.copy()
            best_score = current_best_score
            stats["convergence_counter"] = 0
            if verbose:
                print(f"Generation {generation + 1}: New best solution with score {best_score}")
        else:
            stats["convergence_counter"] += 1
        
        stats["best_score_history"].append(current_best_score)
        stats["avg_score_history"].append(sum(scores) / len(scores))
        
        if termination_condition == "convergence" and stats["convergence_counter"] >= convergence_generations:
            if verbose:
                print(f"Converged after {generation + 1} generations.")
            break
        
        if best_score == 0:
            if verbose:
                print("Perfect solution found!")
            break
        
        next_generation = []
        
        for i in range(min(elite_size, len(population))):
            next_generation.append(population_scores[i][0].copy())
        
        while len(next_generation) < population_size:
            parent1 = tournament_selection(population_scores, tournament_size)
            parent2 = tournament_selection(population_scores, tournament_size)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover_func(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = mutation_func(child1, mutation_rate)
            child2 = mutation_func(child2, mutation_rate)
            
            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)
        
        population = next_generation
    
    if best_solution is None:
        best_solution = current_best_solution.copy()
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score
    
    if verbose:
        print(f"Parallel genetic algorithm completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['generations']} generations")
        print(f"Evaluated {stats['evaluations']} solutions")
        print(f"Best solution has score {best_score}")
    
    return best_solution, stats