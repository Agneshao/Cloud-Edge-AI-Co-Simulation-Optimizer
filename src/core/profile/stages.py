"""Pipeline stage shims for preprocessing, inference, and postprocessing."""

from typing import Any, Callable, Dict, Optional
import numpy as np


class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, name: str, fn: Optional[Callable] = None):
        self.name = name
        self.fn = fn
    
    def run(self, *args, **kwargs) -> Any:
        """Execute the stage."""
        if self.fn:
            return self.fn(*args, **kwargs)
        return None


def create_preprocess_stage(fn: Optional[Callable] = None) -> PipelineStage:
    """Create a preprocessing stage shim."""
    return PipelineStage("preprocess", fn)


def create_inference_stage(fn: Optional[Callable] = None) -> PipelineStage:
    """Create an inference stage shim."""
    return PipelineStage("inference", fn)


def create_postprocess_stage(fn: Optional[Callable] = None) -> PipelineStage:
    """Create a postprocessing stage shim."""
    return PipelineStage("postprocess", fn)

