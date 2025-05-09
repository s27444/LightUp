import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import time
import copy
from tqdm import tqdm
import operator


def protected_division(x, y):
    return x / y if abs(y) > 1e-6 else 1.0


FUNCTION_SET = {
    '+': (operator.add, 2),
    '-': (operator.sub, 2),
    '*': (operator.mul, 2),
    '/': (protected_division, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'exp': (np.exp, 1),
    'log': (lambda x: np.log(abs(x) + 1e-6), 1)
}


class Node:
    def __init__(self, value, arity=0):
        self.value = value
        self.arity = arity
        self.children = []
    
    def is_terminal(self):
        return self.arity == 0
    
    def evaluate(self, variable_values=None):
        if variable_values is None:
            variable_values = {}
        
        if self.is_terminal():
            if isinstance(self.value, (int, float)):
                return float(self.value)
            else:
                return variable_values.get(self.value, 0.0)
        
        func, _ = FUNCTION_SET[self.value]
        args = [child.evaluate(variable_values) for child in self.children]
        return func(*args)
    
    def add_child(self, child):
        if len(self.children) < self.arity:
            self.children.append(child)
            return True
        return False
    
    def depth(self):
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)
    
    def size(self):
        return 1 + sum(child.size() for child in self.children)
    
    def to_string(self):
        if self.is_terminal():
            return str(self.value)
        
        if self.arity == 1:
            return f"{self.value}({self.children[0].to_string()})"
        
        return f"({self.children[0].to_string()} {self.value} {self.children[1].to_string()})"
    
    def __str__(self):
        return self.to_string()


def generate_random_tree(max_depth=4, terminal_ratio=0.5, 
                         variables=None, constants=None, 
                         current_depth=0):
    if variables is None:
        variables = ['x']
    if constants is None:
        constants = [random.uniform(-5, 5) for _ in range(3)]
    
    if current_depth >= max_depth or (current_depth > 0 and random.random() < terminal_ratio):
        if random.random() < 0.5:
            return Node(random.choice(variables))
        else:
            return Node(random.choice(constants))
    
    func_name = random.choice(list(FUNCTION_SET.keys()))
    func, arity = FUNCTION_SET[func_name]
    
    node = Node(func_name, arity)
    
    for _ in range(arity):
        child = generate_random_tree(
            max_depth, terminal_ratio, variables, constants, current_depth + 1
        )
        node.add_child(child)
    
    return node


def crossover(parent1, parent2, max_depth=10):
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    
    def get_all_nodes(node, nodes=None):
        if nodes is None:
            nodes = []
        nodes.append(node)
        for child in node.children:
            get_all_nodes(child, nodes)
        return nodes
    
    nodes1 = get_all_nodes(offspring1)
    nodes2 = get_all_nodes(offspring2)
    
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)
    
    def replace_node(tree, target, replacement):
        if tree is target:
            return copy.deepcopy(replacement)
        
        new_tree = Node(tree.value, tree.arity)
        for child in tree.children:
            new_child = replace_node(child, target, replacement)
            new_tree.add_child(new_child)
        
        return new_tree
    
    new_offspring1 = replace_node(offspring1, crossover_point1, crossover_point2)
    new_offspring2 = replace_node(offspring2, crossover_point2, crossover_point1)
    
    if new_offspring1.depth() <= max_depth and new_offspring2.depth() <= max_depth:
        return new_offspring1, new_offspring2
    
    return copy.deepcopy(parent1), copy.deepcopy(parent2)


def mutation(individual, mutation_rate=0.1, max_depth=2, 
             variables=None, constants=None):
    if variables is None:
        variables = ['x']
    if constants is None:
        constants = [random.uniform(-5, 5) for _ in range(3)]
    
    mutated = copy.deepcopy(individual)
    
    def mutate_subtree(node):
        if random.random() < mutation_rate:
            return generate_random_tree(
                max_depth, 0.5, variables, constants
            )
        
        new_node = Node(node.value, node.arity)
        for child in node.children:
            new_child = mutate_subtree(child)
            new_node.add_child(new_child)
        
        return new_node
    
    return mutate_subtree(mutated)


def evaluate_fitness(individual, x_values, y_values):
    try:
        predictions = []
        for x in x_values:
            variable_values = {'x': x}
            prediction = individual.evaluate(variable_values)
            
            if np.isnan(prediction) or np.isinf(prediction):
                return float('inf')
            
            predictions.append(prediction)
        
        mse = np.mean((np.array(predictions) - y_values) ** 2)
        
        complexity_penalty = 0.01 * individual.size()
        
        return mse + complexity_penalty
    
    except Exception as e:
        return float('inf')


def tournament_selection(population_with_fitness, tournament_size=3):
    tournament = random.sample(population_with_fitness, min(tournament_size, len(population_with_fitness)))
    
    winner, _ = min(tournament, key=lambda x: x[1])
    return winner


def genetic_programming(x_values, y_values, 
                        population_size=100, 
                        max_generations=50,
                        crossover_rate=0.7,
                        mutation_rate=0.1,
                        tournament_size=3,
                        max_depth=5,
                        verbose=False):
    start_time = time.time()
    
    variables = ['x']
    constants = [random.uniform(-5, 5) for _ in range(5)]
    
    population = [
        generate_random_tree(max_depth=4, terminal_ratio=0.5, 
                             variables=variables, constants=constants)
        for _ in range(population_size)
    ]
    
    stats = {
        "best_fitness_history": [],
        "avg_fitness_history": [],
        "best_individual_history": [],
        "execution_time": 0.0,
        "generations": 0
    }
    
    best_individual = None
    best_fitness = float('inf')
    
    generation_iterator = range(max_generations)
    if verbose:
        print(f"Starting genetic programming with population size {population_size}")
        generation_iterator = tqdm(generation_iterator)
    
    for generation in generation_iterator:
        stats["generations"] += 1
        
        population_with_fitness = []
        for individual in population:
            fitness = evaluate_fitness(individual, x_values, y_values)
            population_with_fitness.append((individual, fitness))
        
        population_with_fitness.sort(key=lambda x: x[1])
        
        current_best, current_best_fitness = population_with_fitness[0]
        
        if current_best_fitness < best_fitness:
            best_individual = copy.deepcopy(current_best)
            best_fitness = current_best_fitness
            
            if verbose:
                print(f"Generation {generation + 1}: New best fitness {best_fitness}")
                print(f"Expression: {best_individual}")
        
        fitness_values = [fitness for _, fitness in population_with_fitness]
        stats["best_fitness_history"].append(current_best_fitness)
        stats["avg_fitness_history"].append(np.mean(fitness_values))
        stats["best_individual_history"].append(copy.deepcopy(current_best))
        
        if best_fitness < 1e-6:
            if verbose:
                print("Perfect solution found!")
            break
        
        new_population = []
        
        new_population.append(copy.deepcopy(current_best))
        
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                parent1 = tournament_selection(population_with_fitness, tournament_size)
                parent2 = tournament_selection(population_with_fitness, tournament_size)
                offspring1, offspring2 = crossover(parent1, parent2, max_depth)
                
                offspring1 = mutation(offspring1, mutation_rate, max_depth // 2, variables, constants)
                offspring2 = mutation(offspring2, mutation_rate, max_depth // 2, variables, constants)
                
                new_population.append(offspring1)
                if len(new_population) < population_size:
                    new_population.append(offspring2)
            else:
                parent = tournament_selection(population_with_fitness, tournament_size)
                offspring = mutation(parent, mutation_rate, max_depth // 2, variables, constants)
                new_population.append(offspring)
        
        population = new_population
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_fitness"] = best_fitness
    
    if verbose:
        print(f"Genetic programming completed in {execution_time:.2f} seconds")
        print(f"Best fitness: {best_fitness}")
        print(f"Best expression: {best_individual}")
    
    return best_individual, best_fitness, stats


def run_demo(target_function=None, noise_level=0.1, show_plots=True):
    if target_function is None:
        target_function = lambda x: x**2 + np.sin(x)
    
    x_train = np.linspace(-5, 5, 50)
    y_train = np.array([target_function(x) for x in x_train])
    
    if noise_level > 0:
        y_train += np.random.normal(0, noise_level, size=y_train.shape)
    
    best_individual, best_fitness, stats = genetic_programming(
        x_train, y_train,
        population_size=100,
        max_generations=50,
        crossover_rate=0.7,
        mutation_rate=0.1,
        tournament_size=3,
        max_depth=5,
        verbose=True
    )
    
    x_test = np.linspace(-5, 5, 100)
    y_true = np.array([target_function(x) for x in x_test])
    y_pred = np.array([best_individual.evaluate({'x': x}) for x in x_test])
    
    print("\nResults:")
    print(f"Best expression: {best_individual}")
    print(f"Mean squared error: {best_fitness}")
    
    if show_plots:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_test, y_true, 'b-', label='True function')
        plt.plot(x_test, y_pred, 'r--', label='Evolved expression')
        plt.scatter(x_train, y_train, alpha=0.5, label='Training data')
        plt.title(f'Function Approximation: {best_individual}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(stats["best_fitness_history"], 'b-', label='Best fitness')
        plt.plot(stats["avg_fitness_history"], 'r--', label='Average fitness')
        plt.title('Fitness History')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (MSE)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return best_individual, best_fitness, stats


if __name__ == "__main__":
    print("\n=== Demo 1: x^2 + sin(x) ===")
    run_demo(lambda x: x**2 + np.sin(x))
    
    print("\n=== Demo 2: x^3 - 2*x^2 + x ===")
    run_demo(lambda x: x**3 - 2*x**2 + x)
    
    print("\n=== Demo 3: sin(x) * cos(x/2) ===")
    run_demo(lambda x: np.sin(x) * np.cos(x/2)) 