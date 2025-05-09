from typing import List, Dict, Any
import random
import time
from contextlib import contextmanager


@contextmanager
def timer() -> float:
    start_time = time.time()
    yield start_time
    end_time = time.time()
    return end_time - start_time


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {seconds:.2f}s"


def format_solution_stats(metrics: Dict[str, Any]) -> str:
    is_valid = metrics.get("is_valid", False)
    score = metrics.get("score", float("inf"))
    conflict_count = metrics.get("conflict_count", 0)
    constraint_violation_count = metrics.get("constraint_violation_count", 0)
    unilluminated_count = metrics.get("unilluminated_count", 0)
    bulb_count = metrics.get("bulb_count", 0)
    
    status = "VALID" if is_valid else "INVALID"
    
    return (
        f"Solution Status: {status}\n"
        f"Score: {score}\n"
        f"Light Bulbs: {bulb_count}\n"
        f"Conflicts: {conflict_count}\n"
        f"Constraint Violations: {constraint_violation_count}\n"
        f"Unilluminated Cells: {unilluminated_count}"
    )


def random_choice_with_weights(items: List[Any], weights: List[float]) -> Any:
    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length")
    
    if not items:
        raise ValueError("Items list cannot be empty")
    
    total_weight = sum(weights)
    r = random.uniform(0, total_weight)
    cumulative_weight = 0
    
    for item, weight in zip(items, weights):
        cumulative_weight += weight
        if r <= cumulative_weight:
            return item
    
    return random.choice(items) 