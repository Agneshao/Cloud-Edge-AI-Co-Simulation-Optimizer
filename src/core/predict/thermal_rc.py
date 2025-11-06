"""RC thermal model for time-to-throttle prediction."""

import numpy as np
from typing import Dict, Any


class ThermalRC:
    """1-pole RC thermal model for Jetson devices."""
    
    def __init__(
        self,
        ambient_temp_c: float = 25.0,
        thermal_resistance_c_per_w: float = 0.5,
        thermal_capacitance_j_per_c: float = 10.0,
        max_temp_c: float = 70.0
    ):
        """
        Initialize RC thermal model.
        
        Args:
            ambient_temp_c: Ambient temperature in Celsius
            thermal_resistance_c_per_w: Thermal resistance (R) in °C/W
            thermal_capacitance_j_per_c: Thermal capacitance (C) in J/°C
            max_temp_c: Maximum operating temperature before throttle
        """
        self.ambient_temp = ambient_temp_c
        self.R = thermal_resistance_c_per_w
        self.C = thermal_capacitance_j_per_c
        self.max_temp = max_temp_c
        self.tau = self.R * self.C  # Time constant
    
    def predict_temp(
        self,
        power_w: float,
        time_s: float,
        initial_temp_c: float = None
    ) -> float:
        """
        Predict temperature after time_s seconds at power_w watts.
        
        Args:
            power_w: Power consumption in watts
            time_s: Time duration in seconds
            initial_temp_c: Initial temperature (defaults to ambient)
        
        Returns:
            Predicted temperature in Celsius
        """
        if initial_temp_c is None:
            initial_temp_c = self.ambient_temp
        
        # Steady-state temperature
        steady_state = self.ambient_temp + (power_w * self.R)
        
        # Exponential approach to steady state
        temp = steady_state - (steady_state - initial_temp_c) * np.exp(-time_s / self.tau)
        
        return temp
    
    def time_to_throttle(
        self,
        power_w: float,
        initial_temp_c: float = None
    ) -> float:
        """
        Calculate time to reach throttle temperature.
        
        Args:
            power_w: Power consumption in watts
            initial_temp_c: Initial temperature (defaults to ambient)
        
        Returns:
            Time to throttle in seconds (inf if never reaches max_temp)
        """
        if initial_temp_c is None:
            initial_temp_c = self.ambient_temp
        
        steady_state = self.ambient_temp + (power_w * self.R)
        
        if steady_state <= self.max_temp:
            return float('inf')
        
        # Solve: max_temp = steady_state - (steady_state - initial_temp) * exp(-t/tau)
        # t = -tau * ln((steady_state - max_temp) / (steady_state - initial_temp))
        if steady_state <= initial_temp_c:
            return float('inf')
        
        t = -self.tau * np.log(
            (steady_state - self.max_temp) / (steady_state - initial_temp_c)
        )
        
        return max(0.0, t)

