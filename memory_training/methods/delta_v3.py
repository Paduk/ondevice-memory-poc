"""Delta-v3 profile: Delta-v2 runtime with pending-aware NO_OP sampling."""

from .delta_v2 import DeltaV2Method


class DeltaV3Method(DeltaV2Method):
    """Track the pending-depth training profile without changing runtime semantics."""

    name = "delta_v3"
