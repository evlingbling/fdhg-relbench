"""FDHG relational residual compiler."""

from .ir import CompiledTask, Primitive, PrimitiveFamily, TaskSpec
from .planner import build_candidate_program

__all__ = [
    "CompiledTask",
    "Primitive",
    "PrimitiveFamily",
    "TaskSpec",
    "build_candidate_program",
]
