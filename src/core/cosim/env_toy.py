"""Toy environment for MVP co-simulation.

This is a simple 1D tracking environment that demonstrates how hardware latency
affects task performance. The agent must keep an offset value near zero.
"""

from typing import Dict, Any, Tuple
from .env_base import AbstractEnv


class ToyEnv(AbstractEnv):
    """Simple 1D tracking environment for MVP demonstration.
    
    The environment simulates a tracking task where:
    - offset: Current deviation from target (drifts naturally)
    - action: Correction force to reduce offset
    - done: Episode ends if |offset| > 1.0 (collision/failure)
    
    This demonstrates how hardware latency causes delayed corrections,
    leading to task failure.
    """

    def __init__(self):
        """Initialize the toy environment."""
        self.offset = 0.0
        self.t = 0.0
        self.done = False

    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state.
        
        Returns:
            Initial observation with offset=0, t=0
        """
        self.offset = 0.0
        self.t = 0.0
        self.done = False
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation.
        
        Returns:
            Dictionary with 'offset' and 't' (time)
        """
        return {"offset": self.offset, "t": self.t}

    def step(self, action: float, dt: float) -> Tuple[Dict[str, Any], bool]:
        """Step environment forward by dt seconds.
        
        The dynamics are:
        - Natural drift: offset increases by 0.3 * dt
        - Control correction: offset decreases by 0.5 * action * dt
        
        Args:
            action: Control action (negative values reduce offset)
            dt: Time step in seconds
            
        Returns:
            Tuple of (observation, done) where done=True if |offset| > 1.0
        """
        # Natural drift (simulates external disturbance)
        self.offset += 0.3 * dt
        
        # Control correction (action should be negative to reduce offset)
        self.offset += -0.5 * action * dt
        
        # Update time
        self.t += dt
        
        # Check termination condition (collision/failure)
        if abs(self.offset) > 1.0:
            self.done = True
        
        return self.get_observation(), self.done

