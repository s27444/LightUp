from typing import Tuple, Dict, Any, Optional
from lightup.solution import Solution


def objective_function(solution: Solution) -> Tuple[float, Dict[str, Any]]:
    conflict_count = solution.count_conflict_pairs()
    
    constraint_violations = solution.get_black_cell_constraint_violations()
    
    constraint_violation_count = 0
    zero_constraint_violation_count = 0
    
    for (x, y), (required, actual) in constraint_violations.items():
        if required == 0:
            zero_constraint_violation_count += actual
        else:
            constraint_violation_count += abs(required - actual)
    
    unilluminated = solution.get_unilluminated_cells()
    unilluminated_count = len(unilluminated)
    
    weights = {
        "conflicts": 100,
        "constraint_violations": 1000,
        "zero_constraint_violations": 5000,
        "unilluminated": 1
    }
    
    score = (
        weights["conflicts"] * conflict_count +
        weights["constraint_violations"] * constraint_violation_count +
        weights["zero_constraint_violations"] * zero_constraint_violation_count +
        weights["unilluminated"] * unilluminated_count
    )
    
    metrics = {
        "score": score,
        "conflict_count": conflict_count,
        "constraint_violation_count": constraint_violation_count + zero_constraint_violation_count,
        "zero_constraint_violation_count": zero_constraint_violation_count,
        "unilluminated_count": unilluminated_count,
        "is_valid": solution.is_valid(),
        "bulb_count": len(solution.bulbs)
    }
    
    return score, metrics


def solution_difference(solution1: Solution, solution2: Solution) -> int:
    in1_not_in2 = len(solution1.bulbs - solution2.bulbs)
    in2_not_in1 = len(solution2.bulbs - solution1.bulbs)
    
    return in1_not_in2 + in2_not_in1 