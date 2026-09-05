"""Memory-method adapters and registry."""

from .base import MemoryMethod, ParsedMemoryOutput
from .batch import PatchBatchMethod, SummaryBatchMethod
from .delta import DeltaMethod, DeltaState
from .delta_v2 import DeltaV2Method, DeltaV2State
from .delta_v3 import DeltaV3Method
from .delta_v3_append import DeltaV3AppendMethod
from .delta_v3_compact import (
    DeltaV3CompactK2Method,
    DeltaV3CompactK5Method,
    DeltaV3CompactK10Method,
    DeltaV3CompactMethod,
)
from .patch import PatchMethod
from .general_patch import GeneralPatchMethod
from .summary import SummaryMethod
from .summary_reason import SummaryReasonMethod
from .temporal_patch import TemporalPatchMethod

METHODS = {
    "summary": SummaryMethod,
    "patch": PatchMethod,
    "general_patch": GeneralPatchMethod,
    "delta": DeltaMethod,
    "delta_v2": DeltaV2Method,
    "delta_v3": DeltaV3Method,
    "delta_v3_append": DeltaV3AppendMethod,
    "delta_v3_compact_k2": DeltaV3CompactK2Method,
    "delta_v3_compact_k5": DeltaV3CompactK5Method,
    "delta_v3_compact_k10": DeltaV3CompactK10Method,
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
    "DeltaV3AppendMethod",
    "DeltaV3CompactK2Method",
    "DeltaV3CompactK5Method",
    "DeltaV3CompactK10Method",
    "DeltaV3CompactMethod",
    "DeltaV3Method",
    "MemoryMethod",
    "ParsedMemoryOutput",
    "PatchBatchMethod",
    "PatchMethod",
    "GeneralPatchMethod",
    "SummaryBatchMethod",
    "SummaryMethod",
    "SummaryReasonMethod",
    "TemporalPatchMethod",
]
