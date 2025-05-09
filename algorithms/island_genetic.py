from typing import Dict, Any, List, Tuple, Set, Callable, Optional
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


class Island:
    def __init__(self, 
                board: Board, 
                population_size: int,
                crossover_func: Callable,
                mutation_func: Callable,
                crossover_rate: float,
                mutation_rate: float,
                elite_size: int,
                tournament_size: int,
                island_id: int = 0):
        self.board = board
        self.population_size = population_size
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.island_id = island_id
        
        self.population = [Solution.random_solution(board) for _ in range(population_size)]
        self.evaluated = False
        self.scores = None
        
    def evaluate_population(self) -> List[Tuple[Solution, float]]:
        population_scores = []
        for individual in self.population:
            score, _ = objective_function(individual)
            population_scores.append((individual, score))
        
        population_scores.sort(key=lambda x: x[1])
        self.evaluated = True
        self.scores = [score for _, score in population_scores]
        return population_scores
    
    def evolve(self, population_scores: List[Tuple[Solution, float]]) -> None:
        next_generation = []
        
        for i in range(min(self.elite_size, len(self.population))):
            next_generation.append(population_scores[i][0].copy())
        
        while len(next_generation) < self.population_size:
            parent1 = tournament_selection(population_scores, self.tournament_size)
            parent2 = tournament_selection(population_scores, self.tournament_size)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_func(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            child1 = self.mutation_func(child1, self.mutation_rate)
            child2 = self.mutation_func(child2, self.mutation_rate)
            
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)
        
        self.population = next_generation
        self.evaluated = False
    
    def get_best_individual(self) -> Tuple[Solution, float]:
        if not self.evaluated:
            population_scores = self.evaluate_population()
        else:
            population_scores = list(zip(self.population, self.scores))
            population_scores.sort(key=lambda x: x[1])
        
        return population_scores[0]
    
    def get_migrants(self, count: int) -> List[Solution]:
        if not self.evaluated:
            population_scores = self.evaluate_population()
        else:
            population_scores = list(zip(self.population, self.scores))
            population_scores.sort(key=lambda x: x[1])
        
        migrants = []
        for _ in range(count):
            migrant = tournament_selection(population_scores, self.tournament_size)
            migrants.append(migrant.copy())
        
        return migrants
    
    def receive_migrants(self, migrants: List[Solution]) -> None:
        if not migrants:
            return
        
        if not self.evaluated:
            population_scores = self.evaluate_population()
        else:
            population_scores = list(zip(self.population, self.scores))
            population_scores.sort(key=lambda x: x[1])
        
        for i in range(min(len(migrants), len(self.population))):
            worst_index = len(population_scores) - 1 - i
            if worst_index >= 0:
                self.population[worst_index] = migrants[i]
        
        self.evaluated = False


def solve(board: Board, 
          num_islands: int = 4,
          population_size: int = 50, 
          max_generations: int = 100,
          migration_interval: int = 10,
          migration_rate: float = 0.1,
          crossover_method: str = "uniform", 
          mutation_method: str = "random_flip",
          crossover_rate: float = 0.8, 
          mutation_rate: float = 0.1,
          elite_size: int = 5, 
          tournament_size: int = 3, 
          termination_condition: str = "iterations", 
          convergence_generations: int = 20,
          max_time: float = 300.0, 
          verbose: bool = False,
          distributed: bool = False,
          **kwargs) -> Tuple[Solution, Dict[str, Any]]:
    start_time = time.time()
    
    if verbose:
        print(f"Running island model genetic algorithm with {num_islands} islands")
        print(f"Population size per island: {population_size}")
        print(f"Migration interval: {migration_interval} generations")
        print(f"Migration rate: {migration_rate}")
        print(f"Distributed: {distributed}")
    
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
    
    islands = [
        Island(
            board=board,
            population_size=population_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            island_id=i
        )
        for i in range(num_islands)
    ]
    
    stats = {
        "algorithm": "island_genetic_algorithm",
        "generations": 0,
        "migrations": 0,
        "island_best_scores": [[] for _ in range(num_islands)],
        "island_avg_scores": [[] for _ in range(num_islands)],
        "global_best_scores": [],
        "execution_time": 0.0,
        "islands": num_islands,
        "distributed": distributed,
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
        
        for i, island in enumerate(islands):
            population_scores = island.evaluate_population()
            
            best_score_island = population_scores[0][1]
            avg_score_island = sum(island.scores) / len(island.scores)
            stats["island_best_scores"][i].append(best_score_island)
            stats["island_avg_scores"][i].append(avg_score_island)
            
            if best_score_island < best_score:
                best_solution = population_scores[0][0].copy()
                best_score = best_score_island
                stats["convergence_counter"] = 0
                
                if verbose:
                    print(f"Generation {generation + 1}, Island {i + 1}: New best solution with score {best_score}")
                
                if best_score == 0:
                    if verbose:
                        print("Perfect solution found!")
                    break
            
            island.evolve(population_scores)
        
        stats["global_best_scores"].append(best_score)
        
        if best_score == 0:
            break
        
        if generation > 0 and generation % migration_interval == 0:
            stats["migrations"] += 1
            
            if verbose:
                print(f"Generation {generation + 1}: Performing migration")
            
            migrants_count = max(1, int(population_size * migration_rate))
            
            all_migrants = [island.get_migrants(migrants_count) for island in islands]
            
            for i in range(num_islands):
                next_island = (i + 1) % num_islands
                islands[next_island].receive_migrants(all_migrants[i])
        
        if (termination_condition == "convergence" and 
            stats["convergence_counter"] >= convergence_generations):
            if verbose:
                print(f"Converged after {generation + 1} generations.")
            break
    
    if best_solution is None:
        island_best_solutions = [island.get_best_individual() for island in islands]
        best_solution, best_score = min(island_best_solutions, key=lambda x: x[1])
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score
    
    if verbose:
        print(f"Island model genetic algorithm completed in {execution_time:.2f} seconds")
        print(f"Performed {stats['generations']} generations with {stats['migrations']} migrations")
        print(f"Best solution has score {best_score}")
    
    return best_solution, stats 