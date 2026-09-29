import math

import pytest

from audiojev_inference import AudioInput, Choice, Noul, Score
from audiojev_inference.api import _answer, _prepare, render_prompt
from audiojev_inference.backend import _padded_length, _waveform


def test_jev_shapes_and_candidate_binding():
    choice = Choice("Which event?", {"z": "Zebra", "a": "Apple"})
    kind, instructions, options = _prepare(choice)
    assert options == [("a", "Apple"), ("z", "Zebra")]
    assert "0. Apple" in render_prompt(instructions, options)
    answer = _answer(kind, options, [0.9, 0.1])
    assert answer["choice"] == "a"
    assert answer["probabilities"] == {"a": 0.9, "z": 0.1}
    assert 0 < answer["confidence"] < 1

    kind, _, options = _prepare(Noul("Is there speech?"))
    assert _answer(kind, options, [0.2, 0.8]) == {"type": "noul", "noul": 0.8}

    kind, _, options = _prepare(Score("Quality?", ["poor", "fair", "good"]))
    answer = _answer(kind, options, [0.1, 0.2, 0.7])
    assert math.isclose(answer["score"], 1.6)
    assert answer["legend"] == {"0": "poor", "1": "fair", "2": "good"}
    assert answer["experimental"] is True


def test_relative_choice_preserves_input_order_and_bad_inputs_fail():
    _, _, options = _prepare(Choice("Which?", {"b": "None of the above", "a": "Apple"}))
    assert [key for key, _ in options] == ["b", "a"]
    with pytest.raises(ValueError):
        _prepare(Choice("Which?", {"only": "One"}))
    with pytest.raises(ValueError):
        _prepare(Noul("Yes?", {"maybe": "Unclear"}))
    with pytest.raises(ValueError):
        _prepare(Score("Rating?", ["only one"]))


def test_waveform_input_validation():
    import numpy as np

    samples = _waveform(AudioInput(waveform=np.zeros(800, dtype=np.float32), sample_rate=8000))
    assert samples.dtype == np.float32 and len(samples) == 1600
    with pytest.raises(ValueError):
        _waveform(AudioInput(waveform=np.zeros((100, 2)), sample_rate=16000))
    with pytest.raises(ValueError, match="decodable"):
        _waveform(AudioInput(data=b"not an audio file"))


def test_padding_one_window_past_audio_matches_full_padding():
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor

    # AudioJev's preprocessor: 128 mel bins, padded to 300 s by default.
    extractor = WhisperFeatureExtractor(feature_size=128, chunk_length=300)

    def features(waveform, length):
        return extractor(waveform, sampling_rate=16000, padding="max_length", max_length=length,
                         return_attention_mask=True, return_tensors="pt")

    rng = np.random.default_rng(0)
    # 480,000 samples (30 s) is a hop multiple: without the extra window, STFT edge
    # reflection would leak real audio into its last frame.
    for samples in [400, 160_001, 480_000]:
        waveform = rng.standard_normal(samples).astype(np.float32)
        full = features(waveform, extractor.n_samples)
        short = features(waveform, _padded_length(samples, extractor))
        frames = int(full["attention_mask"].sum())
        assert short["input_features"].shape[-1] < full["input_features"].shape[-1]
        assert int(short["attention_mask"].sum()) == frames
        assert torch.equal(short["input_features"][..., :frames], full["input_features"][..., :frames])
    assert _padded_length(extractor.n_samples + 1, extractor) == extractor.n_samples
