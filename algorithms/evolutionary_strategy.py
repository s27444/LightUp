from typing import Dict, Any, List, Tuple, Callable, Optional
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, dimensions: int, x_range: List[float] = None):
        if x_range is None:
            x_range = [-5.0, 5.0]
        
        self.x = np.random.uniform(x_range[0], x_range[1], dimensions)
        self.sigma = np.ones(dimensions) * 0.5
        self.fitness = None
    
    def mutate(self, tau: float = 0.1, tau_prime: float = 0.1):
        global_factor = np.exp(tau * np.random.normal(0, 1))
        for i in range(len(self.sigma)):
            self.sigma[i] *= global_factor * np.exp(tau_prime * np.random.normal(0, 1))
            self.sigma[i] = max(self.sigma[i], 1e-8)
        
        for i in range(len(self.x)):
            self.x[i] += self.sigma[i] * np.random.normal(0, 1)
    
    def copy(self) -> 'Individual':
        individual = Individual(len(self.x))
        individual.x = np.copy(self.x)
        individual.sigma = np.copy(self.sigma)
        individual.fitness = self.fitness
        return individual


def sphere(x: np.ndarray) -> float:
    return np.sum(x ** 2)


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + a + np.exp(1)


def schwefel(x: np.ndarray) -> float:
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def evolution_strategy(
    fitness_func: Callable[[np.ndarray], float],
    dimensions: int = 10,
    mu: int = 10,
    lambda_: int = 70,
    generations: int = 100,
    plus_selection: bool = False,
    x_range: List[float] = None,
    tau: float = None,
    tau_prime: float = None,
    max_time: float = 60.0,
    verbose: bool = False
) -> Tuple[Individual, Dict[str, Any]]:
    start_time = time.time()
    
    if x_range is None:
        x_range = [-5.0, 5.0]
    
    if tau is None:
        tau = 1.0 / np.sqrt(2 * dimensions)
    if tau_prime is None:
        tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dimensions))
    
    parents = [Individual(dimensions, x_range) for _ in range(mu)]
    
    for ind in parents:
        ind.fitness = fitness_func(ind.x)
    
    parents.sort(key=lambda ind: ind.fitness)
    
    best_individual = parents[0].copy()
    
    stats = {
        "algorithm": "evolution_strategy",
        "variant": "(μ+λ)-ES" if plus_selection else "(μ,λ)-ES",
        "generations": 0,
        "evaluations": mu,
        "best_fitness_history": [best_individual.fitness],
        "avg_fitness_history": [np.mean([ind.fitness for ind in parents])],
        "x_history": [best_individual.x.copy()],
        "sigma_history": [np.mean(best_individual.sigma)],
        "execution_time": 0.0
    }
    
    generation_iterator = range(generations)
    if verbose:
        variant_name = "(μ+λ)-ES" if plus_selection else "(μ,λ)-ES"
        print(f"Running {variant_name} with μ={mu}, λ={lambda_}")
        generation_iterator = tqdm(generation_iterator)
    
    for generation in generation_iterator:
        stats["generations"] += 1
        
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Time limit of {max_time} seconds reached.")
            break
        
        offspring = []
        for _ in range(lambda_):
            parent = random.choice(parents)
            
            child = parent.copy()
            child.mutate(tau, tau_prime)
            
            child.fitness = fitness_func(child.x)
            stats["evaluations"] += 1
            
            offspring.append(child)
        
        if plus_selection:
            population = parents + offspring
        else:
            population = offspring
        
        population.sort(key=lambda ind: ind.fitness)
        parents = [ind.copy() for ind in population[:mu]]
        
        if parents[0].fitness < best_individual.fitness:
            best_individual = parents[0].copy()
            
            if verbose:
                print(f"Generation {generation + 1}: New best fitness {best_individual.fitness}")
                print(f"Best solution: {best_individual.x}")
        
        stats["best_fitness_history"].append(best_individual.fitness)
        stats["avg_fitness_history"].append(np.mean([ind.fitness for ind in parents]))
        stats["x_history"].append(best_individual.x.copy())
        stats["sigma_history"].append(np.mean(best_individual.sigma))
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_fitness"] = best_individual.fitness
    stats["best_solution"] = best_individual.x.copy()
    
    if verbose:
        print(f"Evolution strategy completed in {execution_time:.2f} seconds")
        print(f"Best fitness: {best_individual.fitness}")
        print(f"Best solution: {best_individual.x}")
    
    return best_individual, stats


def run_test_function(function_name: str, 
                      dimensions: int = 10, 
                      generations: int = 100,
                      show_plots: bool = True,
                      save_plots: bool = False,
                      output_dir: str = "results"):
    if function_name == "sphere":
        func = sphere
        x_range = [-5.0, 5.0]
        title = "Sphere Function"
    elif function_name == "rosenbrock":
        func = rosenbrock
        x_range = [-2.0, 2.0]
        title = "Rosenbrock Function"
    elif function_name == "rastrigin":
        func = rastrigin
        x_range = [-5.12, 5.12]
        title = "Rastrigin Function"
    elif function_name == "ackley":
        func = ackley
        x_range = [-32.0, 32.0]
        title = "Ackley Function"
    elif function_name == "schwefel":
        func = schwefel
        x_range = [-500.0, 500.0]
        title = "Schwefel Function"
    else:
        raise ValueError(f"Unknown function: {function_name}")
    
    print(f"\nRunning (μ,λ)-ES on {title}...")
    best_comma, stats_comma = evolution_strategy(
        func, dimensions=dimensions, mu=15, lambda_=100,
        generations=generations, plus_selection=False,
        x_range=x_range, verbose=True
    )
    
    print(f"\nRunning (μ+λ)-ES on {title}...")
    best_plus, stats_plus = evolution_strategy(
        func, dimensions=dimensions, mu=15, lambda_=100,
        generations=generations, plus_selection=True,
        x_range=x_range, verbose=True
    )
    
    print("\nResults:")
    print(f"(μ,λ)-ES: Best fitness = {best_comma.fitness}")
    print(f"(μ+λ)-ES: Best fitness = {best_plus.fitness}")
    
    if show_plots or save_plots:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(stats_comma["best_fitness_history"], 'b-', label='(μ,λ)-ES')
        plt.plot(stats_plus["best_fitness_history"], 'r-', label='(μ+λ)-ES')
        plt.title(f'Convergence on {title} (d={dimensions})')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(stats_comma["sigma_history"], 'b-', label='(μ,λ)-ES')
        plt.plot(stats_plus["sigma_history"], 'r-', label='(μ+λ)-ES')
        plt.title('Step Size Adaptation')
        plt.xlabel('Generation')
        plt.ylabel('Average σ')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{function_name}_d{dimensions}.png")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return best_comma, best_plus, stats_comma, stats_plus


if __name__ == "__main__":
    test_functions = ["sphere", "rosenbrock", "rastrigin", "ackley", "schwefel"]
    dimensions = 10
    
    for func_name in test_functions:
        run_test_function(func_name, dimensions=dimensions, generations=100, show_plots=True) 