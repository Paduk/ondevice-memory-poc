from memory_training.methods.mem0_one_pass import Mem0OnePassMethod


def test_mem0_one_pass_formats_training_row() -> None:
    method = Mem0OnePassMethod()
    row = {
        "input": {"source_batch": [{"text": "Set it to 21 C"}], "retrieved_facts": []},
        "target": {
            "decision": "UPDATE",
            "facts": [{"subject": "Mara", "text": "Temperature is 21 C"}],
            "memory": [
                {"event": "ADD", "subject": "Mara", "text": "Temperature is 21 C"}
            ],
        },
    }
    assert method.format_input(row).startswith('{"source_batch"')
    assert method.format_target(row) == (
        '{"facts":[{"subject":"Mara","text":"Temperature is 21 C"}],'
        '"memory":[{"event":"ADD","subject":"Mara","text":"Temperature is 21 C"}]}'
    )


def test_mem0_one_pass_parse_and_apply() -> None:
    method = Mem0OnePassMethod()
    parsed = method.parse_output(
        '{"facts":[{"subject":"Mara","text":"Temperature is 21 C"}],'
        '"memory":[{"event":"ADD","subject":"Mara","text":"Temperature is 21 C"}]}'
    )
    state = method.apply_output({}, parsed, turn_id="t1")
    assert state == {
        "t1:fact:01": {"subject": "Mara", "text": "Temperature is 21 C"}
    }
    assert method.materialize_memory(state) == (
        "### Mara\n- Temperature is 21 C"
    )


def test_mem0_one_pass_runtime_input_retrieves_from_predicted_state() -> None:
    method = Mem0OnePassMethod(retrieval_k=1)

    class FakeEmbedder:
        def encode(self, texts):
            import numpy as np

            return np.asarray(
                [[1.0, 0.0] if "temperature" in text.lower() else [0.0, 1.0]
                 for text in texts],
                dtype=np.float32,
            )

    method._embedder = FakeEmbedder()
    state = {
        "s081:f0001": {"subject": "Mara", "text": "Temperature is 21 C"},
        "s081:f0002": {"subject": "Mara", "text": "Seat heater is level 2"},
    }
    value = method.runtime_input(
        {
            "input": {
                "source_batch": [
                    {"speaker_name": "Mara", "text": "Change the temperature"}
                ],
                "retrieved_facts": [{"id": "gold-must-not-be-reused"}],
            }
        },
        state,
    )
    assert [fact["id"] for fact in value["retrieved_facts"]] == ["s081:f0001"]


def test_mem0_one_pass_allows_pure_delete_invalidation_fact() -> None:
    method = Mem0OnePassMethod()
    row = {
        "input": {"source_batch": [{"text": "The detour is over"}], "retrieved_facts": []},
        "target": {
            "decision": "UPDATE",
            "facts": [
                {
                    "subject": "shared-vehicle",
                    "text": "The temporary detour is over.",
                    "ephemeral_invalidation": True,
                }
            ],
            "memory": [{"event": "DELETE", "id": "s1:f1"}],
        },
    }
    assert '"ephemeral_invalidation":true' in method.format_target(row)
