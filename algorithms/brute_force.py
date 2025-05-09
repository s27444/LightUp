from typing import Dict, Any, List, Tuple, Set, Optional, Iterator
import time
from tqdm import tqdm
from lightup.board import Board
from lightup.solution import Solution
from lightup.objective import objective_function


def _generate_all_combinations(white_cells: List[Tuple[int, int]], index: int = 0, 
                               current: Optional[Set[Tuple[int, int]]] = None) -> Iterator[Set[Tuple[int, int]]]:
    if current is None:
        current = set()
    
    if index == len(white_cells):
        yield current.copy()
        return
    
    yield from _generate_all_combinations(white_cells, index + 1, current)
    
    current.add(white_cells[index])
    yield from _generate_all_combinations(white_cells, index + 1, current)
    current.remove(white_cells[index])


def solve(board: Board, max_time: float = 60.0, max_solutions: int = 1000000, 
         verbose: bool = False, **kwargs) -> Tuple[Solution, Dict[str, Any]]:
    start_time = time.time()
    white_cells = board.get_white_cells()
    
    best_solution = Solution(board)
    best_score = float("inf")
    
    stats = {
        "algorithm": "brute_force",
        "solutions_checked": 0,
        "valid_solutions_found": 0,
        "execution_time": 0.0,
        "board_size": (board.width, board.height),
        "white_cells_count": len(white_cells),
        "total_combinations": 2 ** len(white_cells)
    }
    
    combinations_iterator = _generate_all_combinations(white_cells)
    if verbose:
        print(f"Total combinations to check: {stats['total_combinations']:,}")
        print(f"Estimated time needed: {stats['total_combinations'] * 0.0001:.2f} seconds")
        print("Starting brute force search...")
        combinations_iterator = tqdm(combinations_iterator, total=min(stats['total_combinations'], max_solutions))
    
    for combination in combinations_iterator:
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Time limit of {max_time} seconds reached.")
            break
        
        if stats["solutions_checked"] >= max_solutions:
            if verbose:
                print(f"Solution limit of {max_solutions} reached.")
            break
        
        solution = Solution(board, set(combination))
        score, metrics = objective_function(solution)
        stats["solutions_checked"] += 1
        
        if metrics["is_valid"]:
            stats["valid_solutions_found"] += 1
        
        if score < best_score:
            best_score = score
            best_solution = solution.copy()
            
            if verbose and stats["solutions_checked"] % 1000 == 0:
                print(f"New best solution found with score {best_score}")
                print(f"Checked {stats['solutions_checked']:,} solutions")
        
        if best_score == 0:
            if verbose:
                print("Perfect solution found!")
            break
    
    execution_time = time.time() - start_time
    stats["execution_time"] = execution_time
    stats["best_score"] = best_score
    
    if verbose:
        print(f"Brute force search completed in {execution_time:.2f} seconds")
        print(f"Checked {stats['solutions_checked']:,} solutions")
        print(f"Found {stats['valid_solutions_found']} valid solutions")
        print(f"Best solution has score {best_score}")
    
    return best_solution, stats