from types import SimpleNamespace

from palmclaw_ubuntu.vehicle_bench.v2_memory_anchors import (
    V2MemoryAnchor,
    V2MemoryAnchorPayload,
    normalize_memory_anchor_evidence,
)


def test_paraphrased_anchor_evidence_is_replaced_with_exact_event_text() -> None:
    event = SimpleNamespace(
        event_id="event-1",
        description="Avery confirms the rear-left reading light at level four.",
    )
    chain = SimpleNamespace(chain_id="chain-1", events=(event,))
    stage2 = SimpleNamespace(
        event_chains=SimpleNamespace(
            payload=SimpleNamespace(vehicle_chains=(chain,))
        )
    )
    payload = V2MemoryAnchorPayload(
        anchors=(
            V2MemoryAnchor(
                chain_id="chain-1",
                source_event_id="event-1",
                source_update_index=0,
                anchor_event_id="event-1",
                evidence_excerpt="Avery confirms the reading light level.",
                reason="This is the first explicit confirmation.",
            ),
        )
    )

    repaired = normalize_memory_anchor_evidence(stage2, payload)

    assert repaired.anchors[0].evidence_excerpt == event.description
