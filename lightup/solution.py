from typing import List, Tuple, Set, Dict, Optional, Callable, Any, Iterator, TypeVar
import random
import copy
import numpy as np
from lightup.board import Board


T = TypeVar('T')


class Solution:
    BLACK_CELL = -1
    BLACK_CELL_ZERO = -2
    _directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    _symbols = {
        'black': '#', 'black_zero': '0', 'black_number': str,
        'bulb_conflict': 'X', 'bulb_valid': 'L',
        'illuminated': '*', 'dark': '.'
    }
    
    def __init__(self, board: Board, bulbs: Optional[Set[Tuple[int, int]]] = None):
        self.board = board
        self.bulbs = set() if bulbs is None else bulbs
    
    def place_bulb(self, x: int, y: int) -> bool:
        if not self.board.is_white_cell(x, y):
            return False
        
        self.bulbs.add((x, y))
        return True
    
    def remove_bulb(self, x: int, y: int) -> bool:
        if (x, y) in self.bulbs:
            self.bulbs.remove((x, y))
            return True
        return False
    
    def _scan_in_direction(self, x: int, y: int, dx: int, dy: int, callback: Callable[[int, int], bool]) -> None:
        max_steps = max(self.board.width, self.board.height)
        
        for step in range(1, max_steps):
            nx, ny = x + dx * step, y + dy * step
            
            if not self.board.is_valid_position(nx, ny):
                break
            
            if self.board.is_black_cell(nx, ny):
                break
            
            if callback(nx, ny):
                break
    
    def _scan_from_positions(self, positions: Iterator[Tuple[int, int]], 
                           process_fn: Callable[[int, int, int, int], None]) -> None:
        for x, y in positions:
            for dx, dy in self._directions:
                process_fn(x, y, dx, dy)
    
    def get_illuminated_cells(self) -> Set[Tuple[int, int]]:
        illuminated = set(self.bulbs)
        
        def process_pos(x: int, y: int, dx: int, dy: int) -> None:
            self._scan_in_direction(x, y, dx, dy, lambda nx, ny: illuminated.add((nx, ny)))
        
        self._scan_from_positions(iter(self.bulbs), process_pos)
        return illuminated
    
    def get_conflicts(self) -> Set[Tuple[int, int]]:
        conflicts = set()
        
        def process_pos(x: int, y: int, dx: int, dy: int) -> None:
            def check_conflict(nx: int, ny: int) -> bool:
                if (nx, ny) in self.bulbs:
                    conflicts.add((x, y))
                    conflicts.add((nx, ny))
                    return True
                return False
            
            self._scan_in_direction(x, y, dx, dy, check_conflict)
        
        self._scan_from_positions(iter(self.bulbs), process_pos)
        return conflicts
    
    def count_conflict_pairs(self) -> int:
        pairs = set()
        
        def process_pos(x: int, y: int, dx: int, dy: int) -> None:
            def count_pair(nx: int, ny: int) -> bool:
                if (nx, ny) in self.bulbs:
                    pair = ((x, y), (nx, ny)) if (x, y) < (nx, ny) else ((nx, ny), (x, y))
                    pairs.add(pair)
                    return True
                return False
            
            self._scan_in_direction(x, y, dx, dy, count_pair)
        
        self._scan_from_positions(iter(self.bulbs), process_pos)
        return len(pairs)
    
    def get_black_cell_constraint_violations(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        def count_adjacent_bulbs(pos: Tuple[int, int]) -> int:
            return sum(1 for nx, ny in self.board.get_neighbors(*pos) if (nx, ny) in self.bulbs)
        
        return {
            pos: (required, count_adjacent_bulbs(pos))
            for pos, required in self.board.get_black_cells().items()
            if count_adjacent_bulbs(pos) != required
        }
    
    def get_unilluminated_cells(self) -> Set[Tuple[int, int]]:
        illuminated = self.get_illuminated_cells()
        return {
            (x, y)
            for y in range(self.board.height)
            for x in range(self.board.width)
            if self.board.is_white_cell(x, y) and (x, y) not in illuminated
        }
    
    def is_valid(self) -> bool:
        return (
            not self.get_conflicts() and
            not self.get_black_cell_constraint_violations() and
            not self.get_unilluminated_cells()
        )
    
    def _is_safe_from_zero_constraint(self, x: int, y: int) -> bool:
        return not any(
            self.board.is_valid_position(nx, ny) and self.board.grid[ny][nx] == self.BLACK_CELL_ZERO
            for nx, ny in self.board.get_neighbors(x, y)
        )
    
    def _find_safe_cells(self, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [cell for cell in cells if self._is_safe_from_zero_constraint(*cell)]
    
    def _filter_for_operation(self, white_cells: List[Tuple[int, int]], 
                           operation: str) -> List[Tuple[int, int]]:
        if operation == 'add':
            empty_cells = [cell for cell in white_cells if cell not in self.bulbs]
            safe_cells = self._find_safe_cells(empty_cells)
            return safe_cells if safe_cells else empty_cells
        elif operation == 'remove':
            return list(self.bulbs)
        return []
    
    def _add_random_bulb(self, solution: "Solution", white_cells: List[Tuple[int, int]]) -> None:
        target_cells = self._filter_for_operation(white_cells, 'add')
        if target_cells:
            x, y = random.choice(target_cells)
            solution.place_bulb(x, y)
    
    def _remove_random_bulb(self, solution: "Solution") -> None:
        if self.bulbs:
            x, y = random.choice(list(self.bulbs))
            solution.remove_bulb(x, y)
    
    def _move_random_bulb(self, solution: "Solution", white_cells: List[Tuple[int, int]]) -> None:
        self._remove_random_bulb(solution)
        self._add_random_bulb(solution, white_cells)
    
    def get_neighbor(self) -> "Solution":
        neighbor = self.copy()
        white_cells = self.board.get_white_cells()
        
        if not self.bulbs and white_cells:
            x, y = random.choice(white_cells)
            neighbor.place_bulb(x, y)
            return neighbor
        
        operations = {
            1: lambda n: self._add_random_bulb(n, white_cells) if white_cells else None,
            2: lambda n: self._remove_random_bulb(n) if self.bulbs else None,
            3: lambda n: self._move_random_bulb(n, white_cells) if self.bulbs and white_cells else None
        }
        
        operation = random.choice(list(operations.keys()))
        if operation in operations:
            operations[operation](neighbor)
        
        if neighbor.bulbs == self.bulbs and white_cells:
            if self.bulbs:
                self._remove_random_bulb(neighbor)
            else:
                self._add_random_bulb(neighbor, white_cells)
        
        return neighbor
    
    def get_neighbors(self, count: int = 10) -> List["Solution"]:
        return [self.get_neighbor() for _ in range(count)]
    
    def get_smart_neighbors(self, count: int = 10) -> List["Solution"]:
        neighbors = []
        problems = self._collect_problems()
        
        remaining = count
        for problem_type, problem_data, weight in problems:
            num_neighbors = min(int(count * weight), remaining)
            
            for _ in range(num_neighbors):
                if remaining <= 0:
                    break
                
                neighbor = self._generate_neighbor_for_problem(problem_type, problem_data)
                if neighbor:
                    neighbors.append(neighbor)
                    remaining -= 1
        
        neighbors.extend(self.get_neighbors(remaining))
        
        return neighbors
    
    def _collect_problems(self) -> List[Tuple[str, Any, float]]:
        problems = []
        
        conflicts = self.get_conflicts()
        black_cell_violations = self.get_black_cell_constraint_violations()
        unilluminated = self.get_unilluminated_cells()
        
        zero_violations = {pos: req_actual for pos, req_actual in black_cell_violations.items() 
                        if req_actual[0] == 0}
        if zero_violations:
            problems.append(('zero', zero_violations, 0.4))
        
        if conflicts:
            problems.append(('conflict', conflicts, 0.2))
        
        non_zero_violations = {pos: req_actual for pos, req_actual in black_cell_violations.items() 
                            if req_actual[0] != 0}
        if non_zero_violations:
            problems.append(('violation', non_zero_violations, 0.2))
        
        if unilluminated:
            problems.append(('unilluminated', unilluminated, 0.2))
        
        return problems
    
    def _generate_neighbor_for_problem(self, problem_type: str, problem_data: Any) -> Optional["Solution"]:
        neighbor = self.copy()
        
        problem_handlers = {
            'zero': self._handle_zero_constraint,
            'conflict': self._handle_conflict,
            'violation': self._handle_black_cell_violation,
            'unilluminated': self._handle_unilluminated
        }
        
        if problem_type in problem_handlers:
            return problem_handlers[problem_type](neighbor, problem_data)
        
        return None
    
    def _handle_zero_constraint(self, neighbor: "Solution", 
                             zero_violations: Dict[Tuple[int, int], Tuple[int, int]]) -> Optional["Solution"]:
        black_cell = random.choice(list(zero_violations.keys()))
        neighbor_cells = self.board.get_neighbors(*black_cell)
        bulb_cells = [(x, y) for x, y in neighbor_cells if (x, y) in self.bulbs]
        
        if bulb_cells:
            x, y = random.choice(bulb_cells)
            neighbor.remove_bulb(x, y)
            return neighbor
        
        return None
    
    def _handle_conflict(self, neighbor: "Solution", conflicts: Set[Tuple[int, int]]) -> Optional["Solution"]:
        conflict_bulb = random.choice(list(conflicts))
        neighbor.remove_bulb(*conflict_bulb)
        return neighbor
    
    def _handle_black_cell_violation(self, neighbor: "Solution", 
                                  violations: Dict[Tuple[int, int], Tuple[int, int]]) -> Optional["Solution"]:
        black_cell = random.choice(list(violations.keys()))
        required, actual = violations[black_cell]
        neighbor_cells = self.board.get_neighbors(*black_cell)
        
        if required > actual:
            empty_cells = [(x, y) for x, y in neighbor_cells 
                         if self.board.is_white_cell(x, y) and (x, y) not in self.bulbs]
            
            if empty_cells:
                safe_cells = self._find_safe_cells(empty_cells)
                target_cells = safe_cells if safe_cells else empty_cells
                
                if target_cells:
                    x, y = random.choice(target_cells)
                    neighbor.place_bulb(x, y)
                    return neighbor
        
        elif required < actual:
            bulb_cells = [(x, y) for x, y in neighbor_cells if (x, y) in self.bulbs]
            
            if bulb_cells:
                x, y = random.choice(bulb_cells)
                neighbor.remove_bulb(x, y)
                return neighbor
        
        return None
    
    def _handle_unilluminated(self, neighbor: "Solution", 
                           unilluminated: Set[Tuple[int, int]]) -> Optional["Solution"]:
        dark_cell = random.choice(list(unilluminated))
        
        if self._is_safe_from_zero_constraint(*dark_cell):
            neighbor.place_bulb(*dark_cell)
            return neighbor
        
        safe_dark_cells = self._find_safe_cells(list(unilluminated))
        
        if safe_dark_cells:
            x, y = random.choice(safe_dark_cells)
            neighbor.place_bulb(x, y)
            return neighbor
        
        return None
    
    @classmethod
    def random_solution(cls, board: Board, density: float = 0.3) -> "Solution":
        solution = cls(board)
        white_cells = board.get_white_cells()
        
        unsafe_cells = set()
        for y in range(board.height):
            for x in range(board.width):
                if board.grid[y][x] == cls.BLACK_CELL_ZERO:
                    for nx, ny in board.get_neighbors(x, y):
                        if board.is_white_cell(nx, ny):
                            unsafe_cells.add((nx, ny))
        
        safe_white_cells = [cell for cell in white_cells if cell not in unsafe_cells]
        cells_to_try = safe_white_cells if safe_white_cells else white_cells
        
        for x, y in cells_to_try:
            if random.random() < density:
                solution.place_bulb(x, y)
        
        return solution
    
    def copy(self) -> "Solution":
        return Solution(self.board, copy.deepcopy(self.bulbs))
    
    def __str__(self) -> str:
        illuminated = self.get_illuminated_cells()
        conflicts = self.get_conflicts()
        
        result = []
        for y in range(self.board.height):
            line = ""
            for x in range(self.board.width):
                cell = self.board.grid[y][x]
                pos = (x, y)
                
                if cell == self.BLACK_CELL:
                    line += self._symbols['black']
                elif cell == self.BLACK_CELL_ZERO:
                    line += self._symbols['black_zero']
                elif cell > 0:
                    line += self._symbols['black_number'](cell)
                elif pos in self.bulbs:
                    line += self._symbols['bulb_conflict'] if pos in conflicts else self._symbols['bulb_valid']
                elif pos in illuminated:
                    line += self._symbols['illuminated']
                else:
                    line += self._symbols['dark']
            result.append(line)
        
        return "\n".join(result)
    
    def __hash__(self) -> int:
        return hash(frozenset(self.bulbs))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return False
        return self.bulbs == other.bulbs
    
    def get_neighbor_normal(self, sigma: float = 0.2) -> "Solution":
        neighbor = self.copy()
        white_cells = self.board.get_white_cells()
        
        if not self.bulbs and white_cells:
            x, y = random.choice(white_cells)
            neighbor.place_bulb(x, y)
            return neighbor
        
        operations_weights = np.array([0.33, 0.33, 0.34])
        
        rand_op = np.random.normal(1.5, sigma)
        rand_op = max(0, min(2.99, rand_op))
        operation = int(rand_op) + 1
        
        operations = {
            1: lambda n: self._add_random_bulb_normal(n, white_cells, sigma) if white_cells else None,
            2: lambda n: self._remove_random_bulb_normal(n, sigma) if self.bulbs else None,
            3: lambda n: self._move_random_bulb_normal(n, white_cells, sigma) if self.bulbs and white_cells else None
        }
        
        if operation in operations:
            operations[operation](neighbor)
        
        if neighbor.bulbs == self.bulbs and white_cells:
            if self.bulbs:
                self._remove_random_bulb(neighbor)
            else:
                self._add_random_bulb(neighbor, white_cells)
        
        return neighbor
    
    def _add_random_bulb_normal(self, solution: "Solution", white_cells: List[Tuple[int, int]], 
                             sigma: float) -> None:
        target_cells = self._filter_for_operation(white_cells, 'add')
        if not target_cells:
            return
        
        if self.bulbs:
            ref_bulb = random.choice(list(self.bulbs))
            
            distances = []
            for cell in target_cells:
                dx = (cell[0] - ref_bulb[0]) / self.board.width
                dy = (cell[1] - ref_bulb[1]) / self.board.height
                distance = (dx**2 + dy**2)**0.5
                distances.append(distance)
            
            max_dist = max(distances) if distances else 1
            normalized_distances = [d/max_dist for d in distances]
            
            weights = [np.exp(-d/sigma) for d in normalized_distances]
            
            if sum(weights) > 0:
                chosen_idx = random.choices(range(len(target_cells)), weights=weights, k=1)[0]
                x, y = target_cells[chosen_idx]
                solution.place_bulb(x, y)
                return
        
        x, y = random.choice(target_cells)
        solution.place_bulb(x, y)
    
    def _remove_random_bulb_normal(self, solution: "Solution", sigma: float) -> None:
        if not self.bulbs:
            return
        
        conflicts = self.get_conflicts()
        
        if conflicts:
            x, y = random.choice(list(conflicts))
            solution.remove_bulb(x, y)
        else:
            constraints = {}
            for x, y in self.bulbs:
                count = sum(1 for nx, ny in self.board.get_neighbors(x, y) 
                          if self.board.is_black_cell(nx, ny))
                constraints[(x, y)] = count
            
            weights = [np.exp(-count/sigma) for count in constraints.values()]
            
            if sum(weights) > 0:
                bulbs_list = list(self.bulbs)
                chosen_idx = random.choices(range(len(bulbs_list)), weights=weights, k=1)[0]
                x, y = bulbs_list[chosen_idx]
                solution.remove_bulb(x, y)
            else:
                x, y = random.choice(list(self.bulbs))
                solution.remove_bulb(x, y)
    
    def _move_random_bulb_normal(self, solution: "Solution", white_cells: List[Tuple[int, int]], 
                              sigma: float) -> None:
        if not self.bulbs:
            return
        
        conflicts = self.get_conflicts()
        if conflicts:
            x, y = random.choice(list(conflicts))
            solution.remove_bulb(x, y)
            self._add_random_bulb_normal(solution, white_cells, sigma)
        else:
            self._remove_random_bulb_normal(solution, sigma)
            self._add_random_bulb_normal(solution, white_cells, sigma)