"""AI-driven search algorithms for configuration optimization.

TODO: Optimization algorithms:
  1. Greedy search (current): Simple, fast, but can get stuck in local optima
     - Try each knob independently, pick best
     - Good starting point for learning
  2. Random search: Randomly sample configurations
     - Simple to implement, sometimes better than greedy
  3. Optuna/Bayesian: More sophisticated, learns from previous trials
     - Better for complex search spaces
     - Requires: pip install optuna
  4. Grid search: Exhaustive search (only for small spaces)
     - Try all combinations
     - Guaranteed to find best, but slow

TODO: Objective function design:
  - What to optimize? (latency, power, accuracy, or combination?)
  - Multi-objective: Use weighted sum or Pareto optimization
  - Constraints: How to handle (hard constraints vs penalties)?
"""

from typing import Dict, Any, Callable, Optional
from .knobs import ConfigKnobs, get_knob_bounds


def greedy_search(
    objective_fn: Callable[[ConfigKnobs], float],
    initial_knobs: Optional[ConfigKnobs] = None,
    max_iterations: int = 50
) -> ConfigKnobs:
    """
    Greedy search for optimal configuration.
    
    TODO: How greedy search works:
      1. Start with initial configuration
      2. Try changing each knob one at a time
      3. Keep the change if it improves the objective
      4. Repeat until no improvement found
      5. Problem: Can get stuck in local optimum (not global best)
    
    TODO: Try these improvements:
      - Add random restarts (start from random configs)
      - Add early stopping (stop if no improvement for N iterations)
      - Add logging (track best score over time)
      - Add parallel search (try multiple configs simultaneously)
    
    Args:
        objective_fn: Function that takes ConfigKnobs and returns objective value (lower is better)
        initial_knobs: Starting configuration
        max_iterations: Maximum search iterations
    
    Returns:
        Best configuration found
    """
    if initial_knobs is None:
        initial_knobs = ConfigKnobs()
    
    best_knobs = initial_knobs
    best_score = objective_fn(best_knobs)
    
    bounds = get_knob_bounds()
    
    for _ in range(max_iterations):
        improved = False
        
        # Try different precision values
        for precision in bounds["precision"]:
            candidate = ConfigKnobs(
                precision=precision,
                resolution=best_knobs.resolution,
                batch_size=best_knobs.batch_size,
                frame_skip=best_knobs.frame_skip,
            )
            score = objective_fn(candidate)
            if score < best_score:
                best_knobs = candidate
                best_score = score
                improved = True
        
        # TODO: Resolution search:
        #   - Current: Fixed grid (limited options)
        #   - Better: Use step size (e.g., 32-pixel increments)
        #   - Best: Use binary search or adaptive sampling
        # Try different resolutions (simplified grid search)
        for h in [320, 480, 640, 960, 1280]:
            for w in [320, 480, 640, 960, 1280]:
                candidate = ConfigKnobs(
                    precision=best_knobs.precision,
                    resolution=(h, w),
                    batch_size=best_knobs.batch_size,
                    frame_skip=best_knobs.frame_skip,
                )
                score = objective_fn(candidate)
                if score < best_score:
                    best_knobs = candidate
                    best_score = score
                    improved = True
        
        # Try different batch sizes
        for batch in range(bounds["batch_size"][0], bounds["batch_size"][1] + 1):
            candidate = ConfigKnobs(
                precision=best_knobs.precision,
                resolution=best_knobs.resolution,
                batch_size=batch,
                frame_skip=best_knobs.frame_skip,
            )
            score = objective_fn(candidate)
            if score < best_score:
                best_knobs = candidate
                best_score = score
                improved = True
        
        if not improved:
            break
    
    return best_knobs

