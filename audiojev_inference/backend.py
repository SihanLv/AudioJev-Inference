"""Qwen2.5-Omni checkpoint loader and audio-tower reuse for local inference."""

from io import BytesIO
import math
from pathlib import Path
import threading

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from .api import LABELS, AudioInput

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _resolve_model_dir(model_dir, *, revision=None, local_files_only=False):
    """Resolve an existing local directory or a versioned Hugging Face snapshot."""
    path = Path(model_dir).expanduser()
    if path.is_dir():
        return path
    if isinstance(model_dir, Path) or path.exists() or str(model_dir).startswith(("/", ".", "~")):
        raise FileNotFoundError(f"model directory does not exist: {path}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=str(model_dir), revision=revision, local_files_only=local_files_only,
        allow_patterns=[
            "config.json", "generation_config.json", "model*.safetensors",
            "model.safetensors.index.json", "tokenizer*.json", "tokenizer.model",
            "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
            "*preprocessor_config.json", "processor_config.json", "chat_template*", "spk_dict.pt",
        ],
    ))


def _waveform(state: AudioInput):
    supplied = sum(value is not None for value in (state.path, state.data, state.waveform))
    if supplied != 1:
        raise ValueError("state needs exactly one of path, data, or waveform")
    if state.waveform is not None:
        samples = np.asarray(state.waveform, dtype=np.float32)
        rate = state.sample_rate
    else:
        source = state.path if state.path is not None else BytesIO(state.data)
        try:
            samples, rate = sf.read(source, dtype="float32", always_2d=True)
        except sf.LibsndfileError as exc:
            raise ValueError("audio data is not a decodable audio file") from exc
        samples = samples.mean(axis=1)
    if not isinstance(rate, int) or rate <= 0 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("invalid mono waveform or sample rate")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if len(samples) < 400 or not np.isfinite(samples).all():
        raise ValueError("audio is too short or contains nonfinite samples")
    return samples


def _padded_length(samples, extractor):
    """Pad one STFT window past the audio rather than to the extractor's 300 s maximum."""
    # Frames past the audio are dropped anyway. A full window of trailing zeros keeps
    # the last kept frame clear of STFT edge reflection, so all kept frames are
    # bit-identical; the cap preserves truncation of longer audio.
    hop = extractor.hop_length
    return min(-(-(samples + extractor.n_fft) // hop) * hop, extractor.n_samples)


class OmniBackend:
    def __init__(self, model_dir, *, adapter=None, device="cuda:0", max_prompt_tokens=4096,
                 merge_adapter=False, revision=None, local_files_only=False):
        import torch
        from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("AudioJev Qwen2.5-Omni inference requires an explicit CUDA device")
        if self.device.index is None:
            raise ValueError("device must include its CUDA index, such as cuda:0")
        if not isinstance(max_prompt_tokens, int) or max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        model_dir = _resolve_model_dir(model_dir, revision=revision, local_files_only=local_files_only)
        self.max_prompt_tokens = max_prompt_tokens
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        torch.cuda.set_device(self.device)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_dir, local_files_only=True, use_fast=False)
        token_ids = [self.processor.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if any(len(ids) != 1 for ids in token_ids) or len({ids[0] for ids in token_ids}) != 36:
            raise ValueError("model tokenizer does not have 36 distinct single-token labels")
        self.label_ids = [ids[0] for ids in token_ids]
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        config.enable_audio_output = False
        wrapper = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_dir, config=config, local_files_only=True, use_safetensors=True,
            dtype=torch.bfloat16, device_map={"": self.device.index},
            low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
        self.wrapper = wrapper
        self.model = wrapper.thinker
        if adapter:
            from peft import PeftModel
            if not (Path(adapter) / "adapter_model.safetensors").is_file():
                raise FileNotFoundError("adapter_model.safetensors is missing")
            self.model = PeftModel.from_pretrained(
                self.model, adapter, is_trainable=False, local_files_only=True).eval()
            if merge_adapter:
                self.model = self.model.merge_and_unload(safe_merge=True).eval()
                wrapper.thinker = self.model
        elif merge_adapter:
            raise ValueError("merge_adapter requires an adapter")
        self.model.requires_grad_(False)
        self.base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self._lock = threading.Lock()
        self._audio_features = None
        self._encoder_calls = 0
        original = self.base.get_audio_features

        def cached_features(*args, **kwargs):
            if self._audio_features is not None:
                return self._audio_features
            value = original(*args, **kwargs)
            self._encoder_calls += 1
            self._audio_features = value.detach()
            return value

        self.base.get_audio_features = cached_features

    def predict_many(self, state, prompts):
        torch = self.torch
        waveform = _waveform(state)
        padded = _padded_length(len(waveform), self.processor.feature_extractor)
        results = {}
        prompt_tokens = 0
        # The wrapped audio encoder has request-scoped mutable state.
        with self._lock:
            self._audio_features = None
            calls_before = self._encoder_calls
            try:
                for key, prompt, count in prompts:
                    content = [{"type": "audio", "audio": "audio.wav"},
                               {"type": "text", "text": prompt}]
                    conversation = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": content},
                    ]
                    text = self.processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False)
                    # With audio_kwargs, top-level audio options are ignored, and the
                    # processor empties the dict: build it per call.
                    batch = self.processor(text=[text], audio=[waveform], return_tensors="pt",
                                           padding=True, use_audio_in_video=False,
                                           audio_kwargs={"sampling_rate": 16000, "max_length": padded})
                    if batch["input_ids"].shape[1] > self.max_prompt_tokens:
                        raise ValueError("prompt exceeds max_prompt_tokens; no silent truncation")
                    frames = int(batch["feature_attention_mask"].sum(-1).max())
                    batch["input_features"] = batch["input_features"][:, :, :frames]
                    batch["feature_attention_mask"] = batch["feature_attention_mask"][:, :frames]
                    batch = {name: value.to(self.device) if torch.is_tensor(value) else value
                             for name, value in batch.items()}
                    prompt_tokens += int(batch["input_ids"].shape[1])
                    # The tested research readout projects only the final decision
                    # position; this avoids the full sequence-vocabulary matrix.
                    hook = self.base.lm_head.register_forward_pre_hook(
                        lambda _module, args: (args[0][:, -1:, :],))
                    try:
                        with torch.inference_mode():
                            logits = self.model(**batch, use_cache=False, return_dict=True).logits[0, -1].float()
                    finally:
                        hook.remove()
                    selected = logits[self.label_ids[:count]]
                    values = torch.softmax(selected, dim=-1)
                    if not bool(torch.isfinite(values).all()):
                        raise RuntimeError("nonfinite candidate probabilities")
                    results[key] = values.cpu().tolist()
            finally:
                self._audio_features = None
        return results, {"input_tokens": prompt_tokens, "output_tokens": 0,
                         "audio_encoder_calls": self._encoder_calls - calls_before,
                         "questions": len(prompts)}
