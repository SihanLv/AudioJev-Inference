import pytest

from audiojev_inference import DEFAULT_MODEL, AudioJev
from audiojev_inference.backend import _resolve_model_dir


def test_local_model_resolution_never_downloads(tmp_path, monkeypatch):
    def unexpected_download(**kwargs):
        pytest.fail("a local model path must not contact the Hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)
    assert _resolve_model_dir(tmp_path) == tmp_path
    assert _resolve_model_dir(str(tmp_path)) == tmp_path
    for missing in [tmp_path / "missing", str(tmp_path / "missing"), "./missing-model"]:
        with pytest.raises(FileNotFoundError):
            _resolve_model_dir(missing)


def test_hub_model_honors_revision_and_offline_mode(tmp_path, monkeypatch):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    assert _resolve_model_dir(DEFAULT_MODEL, revision="release-sha", local_files_only=True) == tmp_path
    assert calls[0]["repo_id"] == "shlv/AudioJev"
    assert calls[0]["revision"] == "release-sha"
    assert calls[0]["local_files_only"] is True


def test_python_api_forwards_model_source_options(monkeypatch):
    calls = []

    class Backend:
        def __init__(self, model_dir, **kwargs):
            calls.append((model_dir, kwargs))

    monkeypatch.setattr("audiojev_inference.backend.OmniBackend", Backend)
    AudioJev(revision="release-sha", local_files_only=True)
    assert calls[0][0] == DEFAULT_MODEL
    assert calls[0][1]["revision"] == "release-sha"
    assert calls[0][1]["local_files_only"] is True
