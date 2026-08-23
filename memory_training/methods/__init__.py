"""Memory-method adapters and registry."""

from .base import MemoryMethod, ParsedMemoryOutput
from .delta import DeltaMethod, DeltaState
from .patch import PatchMethod
from .summary import SummaryMethod
from .summary_reason import SummaryReasonMethod

METHODS = {
    "summary": SummaryMethod,
    "patch": PatchMethod,
    "delta": DeltaMethod,
    "summary_reason": SummaryReasonMethod,
}

__all__ = [
    "METHODS",
    "DeltaMethod",
    "DeltaState",
    "MemoryMethod",
    "ParsedMemoryOutput",
    "PatchMethod",
    "SummaryMethod",
    "SummaryReasonMethod",
]
