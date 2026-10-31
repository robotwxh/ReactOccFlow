"""Research-safe candidate generation and offline planning contracts.

The implementation is deterministic and CPU friendly. It scores candidate paths; it
makes no claim about a trained flow-matching model or closed-loop driving quality.
"""

from .planflow import (
    Candidate,
    CandidateGenerator,
    CandidatePlans,
    CostBreakdown,
    KinematicCheck,
    PlanFlow,
    Planner,
    PlanningConfig,
    SelectionResult,
    check_kinematics,
    compute_costs,
)

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "CandidatePlans",
    "CostBreakdown",
    "KinematicCheck",
    "PlanFlow",
    "Planner",
    "PlanningConfig",
    "SelectionResult",
    "check_kinematics",
    "compute_costs",
]
