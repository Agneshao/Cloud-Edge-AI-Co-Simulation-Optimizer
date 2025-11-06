"""Configuration knobs for optimization."""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ConfigKnobs:
    """Configuration knobs for model optimization."""
    
    precision: str = "FP16"  # INT8, FP16, FP32
    resolution: tuple[int, int] = (640, 480)  # (height, width)
    batch_size: int = 1
    frame_skip: int = 0  # Process every (frame_skip + 1) frame
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "frame_skip": self.frame_skip,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigKnobs":
        """Create from dictionary."""
        return cls(
            precision=data.get("precision", "FP16"),
            resolution=tuple(data.get("resolution", (640, 480))),
            batch_size=data.get("batch_size", 1),
            frame_skip=data.get("frame_skip", 0),
        )


def get_knob_bounds() -> Dict[str, tuple]:
    """Get bounds for each knob."""
    return {
        "precision": ("INT8", "FP16", "FP32"),
        "resolution_height": (320, 1280),
        "resolution_width": (320, 1280),
        "batch_size": (1, 8),
        "frame_skip": (0, 4),
    }

