from memory_training.methods.mem0_two_stage import Mem0TwoStageMethod
from memory_training import validation


def test_two_stage_formats_both_tasks() -> None:
    method = Mem0TwoStageMethod()
    extraction = {
        "task_type": "EXTRACT",
        "input": {"source_batch": [{"text": "Set it to 21 C"}]},
        "target": {
            "decision": "UPDATE",
            "facts": [{"subject": "Mara", "text": "Temperature is 21 C"}],
        },
    }
    manager = {
        "task_type": "MANAGE",
        "input": {
            "new_facts": [{"subject": "Mara", "text": "Temperature is 21 C"}],
            "retrieved_facts": [],
        },
        "target": {
            "decision": "UPDATE",
            "memory": [{"event": "ADD", "subject": "Mara", "text": "Temperature is 21 C"}],
        },
    }
    assert '"task":"EXTRACT"' in method.format_input(extraction)
    assert method.format_target(extraction).startswith('{"facts"')
    assert '"task":"MANAGE"' in method.format_input(manager)
    assert method.format_target(manager).startswith('{"memory"')


def test_two_stage_runtime_builds_extract_then_predicted_retrieval() -> None:
    method = Mem0TwoStageMethod(retrieval_k=1)

    class FakeEmbedder:
        def encode(self, texts):
            import numpy as np

            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    method.retriever._embedder = FakeEmbedder()
    state = {
        "s081:f0001": {"subject": "Mara", "text": "Temperature is 20 C"}
    }
    extract = method.runtime_row(
        {"current_turn": {"speaker_name": "Mara", "text": "Set it to 21 C"}},
        state,
    )
    assert extract["task_type"] == "EXTRACT"
    manage = method.manager_row(
        extract, [{"subject": "Mara", "text": "Temperature is 21 C"}]
    )
    assert manage["task_type"] == "MANAGE"
    assert manage["input"]["retrieved_facts"][0]["id"] == "s081:f0001"


def test_two_stage_generation_skips_manager_for_empty_facts(monkeypatch) -> None:
    method = Mem0TwoStageMethod()
    rows = [
        {"task_type": "EXTRACT", "input": {}, "_runtime_state": {}},
        {"task_type": "EXTRACT", "input": {}, "_runtime_state": {}},
    ]
    calls = []

    def fake_generate(*args, **kwargs):
        batch = args[3]
        calls.append(batch)
        if batch[0]["task_type"] == "EXTRACT":
            return [
                ('{"facts":[]}', 10, 2, 0.1),
                ('{"facts":[{"subject":"Mara","text":"Temperature is 21 C"}]}',
                 11, 7, 0.2),
            ]
        return [('{"memory":[{"event":"ADD","subject":"Mara",'
                 '"text":"Temperature is 21 C"}]}', 20, 9, 0.3)]

    monkeypatch.setattr(validation, "_generate_outputs_batch_once", fake_generate)
    results = validation._generate_two_stage_outputs_batch(
        None,
        None,
        method,
        rows,
        encoder=None,
        accelerator=None,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    assert len(calls) == 2
    assert results[0] == ('{"memory":[]}', 10, 2, 0.1)
    assert results[1][1:] == (31, 16, 0.5)
