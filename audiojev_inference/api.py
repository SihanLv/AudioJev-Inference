"""Jev-shaped typed decisions from an audio input."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_MODEL = "shlv/AudioJev"
_RELATIVE = re.compile(
    r"\b(?:all|none|both|neither|any)\s+of\s+the\s+(?:above|below)\b|"
    r"\b(?:option|choice)\s+[A-Z0-9]\b|\b(?:former|latter)\b", re.I
)


@dataclass(frozen=True)
class AudioInput:
    """A local audio path, encoded audio bytes, or mono 16 kHz waveform."""

    path: str | Path | None = None
    data: bytes | None = None
    waveform: object | None = None
    sample_rate: int = 16000


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[str] | tuple[str, ...]


def _description(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("criteria descriptions must be nonempty strings")
    return value.strip()


def _prepare(question):
    if isinstance(question, Mapping):
        kind = question.get("type")
        if kind == "choice":
            question = Choice(question.get("instructions"), question.get("criteria"))
        elif kind == "noul":
            question = Noul(question.get("instructions"), question.get("criteria"))
        elif kind == "score":
            question = Score(question.get("instructions"), question.get("criteria"))
        else:
            raise ValueError(f"unsupported question type: {kind!r}")
    if not isinstance(question, (Choice, Noul, Score)):
        raise ValueError("question must be Choice, Noul, or Score")
    instructions = _description(question.instructions)
    if isinstance(question, Choice):
        if not isinstance(question.criteria, Mapping):
            raise ValueError("choice criteria must be a mapping")
        options = [(key, _description(value)) for key, value in question.criteria.items()]
        if any(not isinstance(key, str) or not key for key, _ in options):
            raise ValueError("choice keys must be nonempty strings")
        kind = "choice"
    elif isinstance(question, Noul):
        criteria = question.criteria or {}
        if not isinstance(criteria, Mapping) or set(criteria) - {"false", "true"}:
            raise ValueError("noul criteria may contain only 'true' and 'false'")
        options = [("false", _description(criteria.get("false", "No; the statement is false"))),
                   ("true", _description(criteria.get("true", "Yes; the statement is true")))]
        kind = "noul"
    else:
        if not isinstance(question.criteria, (list, tuple)):
            raise ValueError("score criteria must be an ordered list")
        options = [(str(i), _description(value)) for i, value in enumerate(question.criteria)]
        kind = "score"
    if not 2 <= len(options) <= (10 if kind == "score" else 36):
        raise ValueError("requires 2–10 score levels or 2–36 choice options")
    if kind in ("choice", "noul"):
        texts = [text for _, text in options]
        if len(set(texts)) == len(texts) and not any(_RELATIVE.search(text) for text in texts):
            options.sort(key=lambda item: item[1].encode("utf-8"))
    return kind, instructions, options


def render_prompt(instructions, options):
    lines = ["You are a structured decision module.",
             "Use the audio and the option descriptions as evidence.",
             "Output exactly one listed answer label token and nothing else.",
             "Do not output an explanation, punctuation, another word, or a label not listed below.",
             "There is no implicit other/none option; choose among the listed options.",
             instructions, "", "Options:"]
    lines.extend(f"{LABELS[i]}. {text}" for i, (_, text) in enumerate(options))
    lines.extend(["", f"Reply with exactly one option label ({', '.join(LABELS[:len(options)])}).", "Answer:"])
    return "\n".join(lines)


def _concentration(probabilities):
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def _answer(kind, options, probabilities):
    values = {key: float(value) for (key, _), value in zip(options, probabilities)}
    if kind == "noul":
        return {"type": "noul", "noul": values["true"]}
    confidence = _concentration(probabilities)
    if kind == "choice":
        winner = max(range(len(options)), key=lambda i: probabilities[i])
        return {"type": "choice", "choice": options[winner][0],
                "probabilities": values, "confidence": confidence}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probabilities)),
            "probabilities": values, "legend": {key: text for key, text in options},
            "confidence": confidence, "experimental": True}


class AudioJev:
    """Keep one checkpoint resident and answer multiple questions per audio clip."""

    def __init__(self, model_dir=DEFAULT_MODEL, *, adapter=None, device="cuda:0", max_prompt_tokens=4096,
                 merge_adapter=False, revision=None, local_files_only=False):
        from .backend import OmniBackend
        self.backend = OmniBackend(model_dir, adapter=adapter, device=device,
                                   max_prompt_tokens=max_prompt_tokens, merge_adapter=merge_adapter,
                                   revision=revision, local_files_only=local_files_only)
        self.model = "audiojev-local"

    def system_one(self, *, state, questions: Mapping[str, object]):
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        prepared = {}
        for key, question in questions.items():
            if not isinstance(key, str) or not key:
                raise ValueError("question ids must be nonempty strings")
            prepared[key] = _prepare(question)
        if isinstance(state, (str, Path)):
            state = AudioInput(path=state)
        elif isinstance(state, bytes):
            state = AudioInput(data=state)
        if not isinstance(state, AudioInput):
            raise ValueError("state must be an audio path, bytes, or AudioInput")
        prompts = [(key, render_prompt(instructions, options), len(options))
                   for key, (_, instructions, options) in prepared.items()]
        predictions, usage = self.backend.predict_many(state, prompts)
        answers = {key: _answer(kind, options, predictions[key])
                   for key, (kind, _, options) in prepared.items()}
        return {"model": self.model, "answers": answers, "usage": usage,
                "confidence_definition": "1 - normalized entropy; distribution concentration, not correctness probability"}
