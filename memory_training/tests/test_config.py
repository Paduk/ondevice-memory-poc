import argparse
import os
from pathlib import Path

from pytest import MonkeyPatch

from memory_training.config import MODEL_BY_KEY, split_for_scenario
from memory_training.train import _configure_runtime_paths, _loader_worker_kwargs


def test_scenario_split_boundaries() -> None:
    assert split_for_scenario(1) == "train"
    assert split_for_scenario(80) == "train"
    assert split_for_scenario(81) == "validation"
    assert split_for_scenario(90) == "validation"
    assert split_for_scenario(91) == "test"
    assert split_for_scenario(100) == "test"
    assert split_for_scenario(101) == "train"
    assert split_for_scenario(110) == "train"
    assert split_for_scenario(111) == "validation"
    assert split_for_scenario(112) == "test"
    assert split_for_scenario(120) == "test"
    assert split_for_scenario(201) == "train"
    assert split_for_scenario(250) == "train"
    assert split_for_scenario(301) == "train"
    assert split_for_scenario(320) == "train"


def test_qwen35_2b_target_is_registered() -> None:
    target = MODEL_BY_KEY["qwen3.5-2b"]
    assert target.parameters_b == 2
    assert target.hf_id == "Qwen/Qwen3.5-2B"


def test_qwen35_0_8b_target_is_registered() -> None:
    target = MODEL_BY_KEY["qwen3.5-0.8b"]
    assert target.parameters_b == 0.8
    assert target.hf_id == "Qwen/Qwen3.5-0.8B"


def test_granite4_350m_target_is_registered() -> None:
    target = MODEL_BY_KEY["granite4-350m"]
    assert target.parameters_b == 0.35
    assert target.hf_id == "ibm-granite/granite-4.0-350m"


def test_granite4_1b_target_is_registered() -> None:
    target = MODEL_BY_KEY["granite4-1b"]
    assert target.parameters_b == 1
    assert target.hf_id == "ibm-granite/granite-4.0-1b"


def test_training_runtime_paths_override_xdg_cache(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", "/wrong/huggingface")
    monkeypatch.setenv("XDG_CACHE_HOME", "/wrong/xdg")

    _configure_runtime_paths(tmp_path)

    expected = tmp_path / "cache" / "huggingface"
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert os.environ["HF_HOME"] == str(expected)
    assert os.environ["HF_HUB_CACHE"] == str(expected / "hub")
    assert os.environ["HF_XET_CACHE"] == str(expected / "xet")
    assert os.environ["TMPDIR"] == str(tmp_path / "tmp")


def test_loader_workers_enable_prefetch_only_for_multiprocessing() -> None:
    enabled = argparse.Namespace(num_workers=4, pin_memory=True, prefetch_factor=2)
    disabled = argparse.Namespace(num_workers=0, pin_memory=False, prefetch_factor=2)

    assert _loader_worker_kwargs(enabled) == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    assert _loader_worker_kwargs(disabled) == {
        "num_workers": 0,
        "pin_memory": False,
    }
