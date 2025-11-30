"""Abstract base class for co-simulation environments."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class AbstractEnv(ABC):
    """Abstract interface for co-simulation environments.
    
    This interface allows the co-simulation loop to work with any environment
    implementation (ToyEnv, IsaacSimEnv, RealRobotEnv) without modification.
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state.
        
        Returns:
            Initial observation dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from the environment.
        
        Returns:
            Current observation dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: float, dt: float) -> Tuple[Dict[str, Any], bool]:
        """Step the environment forward by dt seconds.
        
        Args:
            action: Control action (e.g., correction force)
            dt: Time step in seconds
            
        Returns:
            Tuple of (observation, done) where done indicates if episode terminated
        """
        raise NotImplementedError

