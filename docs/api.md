# API

**English** | [简体中文](api.zh-CN.md) · [Home](../README.md)

## Load a model

```python
from audiojev_inference import AudioJev

model = AudioJev(device="cuda:0")
```

By default, the library loads [shlv/AudioJev](https://huggingface.co/shlv/AudioJev), downloading and caching the required files automatically. You can also specify a model ID or local directory:

```python
model = AudioJev("./models/AudioJev", device="cuda:0", local_files_only=True)
```

| Parameter | Default | Description |
|---|---|---|
| `model_dir` | `shlv/AudioJev` | Model ID or local directory |
| `device` | `cuda:0` | CUDA device |
| `revision` | `None` | Model branch, tag, or commit SHA; uses the main branch when omitted |
| `local_files_only` | `False` | Load only local files or an existing cached model |
| `max_prompt_tokens` | `4096` | Maximum number of tokens in the complete input |

Model instances are reusable. Create one at application startup and reuse it for subsequent requests.

## Audio input

`model.system_one(state=..., questions=...)` accepts one audio clip and a mapping of questions.

| `state` type | Example |
|---|---|
| File path | `"./example.wav"` or `Path("./example.wav")` |
| Encoded audio file bytes | `Path("./example.wav").read_bytes()` |
| Waveform | `AudioInput(waveform=mono_array, sample_rate=16000)` |

```python
from audiojev_inference import AudioInput

audio = AudioInput(waveform=mono_array, sample_rate=16000)
```

File and byte inputs support formats that SoundFile can decode, such as WAV and FLAC. Multichannel files are mixed to mono and resampled to 16 kHz. Waveforms supplied directly must be one-dimensional arrays with a positive integer sample rate. Supply one audio source at a time.

## Choice: candidate selection

```python
from audiojev_inference import Choice

question = Choice(
    instructions="Which event is audible?",
    criteria={"dog": "A dog barking", "horn": "A car horn"},
)
```

`criteria` maps 2–36 candidate keys to descriptions. Keys identify candidates in the response; descriptions are used by the model. Both must be nonempty strings.

Example response:

```json
{
  "type": "choice",
  "choice": "dog",
  "probabilities": {"dog": 0.8, "horn": 0.2},
  "confidence": 0.2781
}
```

`choice` is the selected candidate's key, and `probabilities` contains the probability of every candidate. The numbers above illustrate the response format.

## Noul: yes/no judgment

```python
from audiojev_inference import Noul

question = Noul(instructions="Can speech be heard in this clip?")
```

Returns a response such as `{"type": "noul", "noul": 0.8}`, where `noul` is the probability that the statement is true.

Use `criteria={"false": "...", "true": "..."}` to customize the descriptions of the two outcomes.

## Score: ordinal scoring

```python
from audiojev_inference import Score

question = Score(
    instructions="Rate speech clarity.",
    criteria=["Unclear", "Partly clear", "Clear"],
)
```

`criteria` contains 2–10 descriptions in level order. The response includes `score`, `probabilities`, `legend`, `confidence`, and `experimental: true`.

Level indices start at 0, and `score` is their probability-weighted expectation. For example, probabilities of `[0.1, 0.2, 0.7]` across three levels give a score of `1.6`. Score is an experimental interface; the model has not been specifically trained for arbitrary human rating scales.

## Probabilities and candidate descriptions

Probabilities are normalized over the supplied candidates. If you need an “other” answer, add it explicitly to `criteria`.

`confidence = 1 - H(p)/log(K)` measures distribution concentration on a scale from 0 to 1; it is not prediction accuracy.

Choice/Noul sorts unique descriptions by their UTF-8 bytes when no recognized position-dependent wording is present, then maps results back to the original keys. Duplicate descriptions or phrases such as `all/none of the above` and `option A` preserve the input order. Self-contained candidate descriptions make the question easier to specify. Score always preserves level order.

## Responses to multiple questions

The keys in `questions` are caller-defined question IDs. `system_one` returns answers under those IDs:

```python
result = model.system_one(
    state="./example.wav",
    questions={"event": question},
)
answer = result["answers"]["event"]
```

| Top-level field | Description |
|---|---|
| `model` | Service model identifier, `audiojev-local` |
| `answers` | Mapping from question IDs to answers |
| `usage` | Input token count, question count, and audio encoder call count |
| `confidence_definition` | Definition of the `confidence` field |

`usage` contains `input_tokens`, `output_tokens`, `audio_encoder_calls`, and `questions`. The interface computes candidate probabilities directly, so `output_tokens` is 0. Multiple questions in one request share one audio encoding.

## HTTP interface

### POST /v1/systemone

```json
{
  "state": {"audio_base64": "<base64 encoded WAV>"},
  "questions": {
    "sound": {
      "type": "choice",
      "instructions": "Which event is audible?",
      "criteria": {"dog": "A dog barking", "horn": "A car horn"}
    },
    "speech": {
      "type": "noul",
      "instructions": "Can speech be heard?"
    }
  }
}
```

HTTP requests supply audio through `state.audio_base64`. Question objects use `type: choice | noul | score`; their other fields match the corresponding Python types. Responses have the same structure as `system_one` responses.

The `model` field is optional. If supplied, its value must be `audiojev-local`.

### GET /health

Once the model has loaded, returns:

```json
{"status": "ok", "model": "audiojev-local"}
```

Full interactive documentation is available at `/docs`.

### Input limits

- The audio file must be at most 20,000,000 bytes after base64 decoding.
- The complete prompt, including audio, the question, and candidates, is limited to 4,096 tokens.
- Audio must contain at least 25 ms of valid samples.

Malformed requests, invalid audio, and inputs exceeding these limits return HTTP 422, with the reason in `detail`.
