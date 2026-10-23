"""Portable archetypal analysis with explicit, versioned solver profiles."""

from .artifacts import SavedFit, load_fit, transform
from .solver import SOLVER_CONTRACT_VERSION, SOLVER_ID

__all__ = ["SOLVER_CONTRACT_VERSION", "SOLVER_ID", "SavedFit", "load_fit", "transform"]
__version__ = "0.3.0"
