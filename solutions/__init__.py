"""Reference solutions for the two C12 hackathon challenges."""

from .detection import DetectionResult, InterdotDetector, pixel_metrics
from .optimization import ContrastOptimizer, OptimizationResult, optimize

__all__ = [
    "ContrastOptimizer",
    "DetectionResult",
    "InterdotDetector",
    "OptimizationResult",
    "optimize",
    "pixel_metrics",
]
