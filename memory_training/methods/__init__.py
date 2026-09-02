"""Memory-method adapters and registry."""

from .base import MemoryMethod, ParsedMemoryOutput
from .batch import PatchBatchMethod, SummaryBatchMethod
from .delta import DeltaMethod, DeltaState
from .delta_v2 import DeltaV2Method, DeltaV2State
from .delta_v3 import DeltaV3Method
from .patch import PatchMethod
from .summary import SummaryMethod
from .summary_reason import SummaryReasonMethod
from .temporal_patch import TemporalPatchMethod

METHODS = {
    "summary": SummaryMethod,
    "patch": PatchMethod,
    "delta": DeltaMethod,
    "delta_v2": DeltaV2Method,
    "delta_v3": DeltaV3Method,
    "summary_reason": SummaryReasonMethod,
    "summary_batch": SummaryBatchMethod,
    "patch_batch": PatchBatchMethod,
    "temporal_patch": TemporalPatchMethod,
}

__all__ = [
    "METHODS",
    "DeltaMethod",
    "DeltaState",
    "DeltaV2Method",
    "DeltaV2State",
    "DeltaV3Method",
    "MemoryMethod",
    "ParsedMemoryOutput",
    "PatchBatchMethod",
    "PatchMethod",
    "SummaryBatchMethod",
    "SummaryMethod",
    "SummaryReasonMethod",
    "TemporalPatchMethod",
]
