# AudioJev-Inference

**English** | [简体中文](README.zh-CN.md)

A Python inference library and HTTP service for AudioJev. Provide an audio clip and natural-language questions to get candidate probabilities, yes/no judgments, or ordinal scores. A single request can ask multiple questions about the same audio.

Model: [shlv/AudioJev](https://huggingface.co/shlv/AudioJev) · [API reference](docs/api.md) · [Deployment guide](docs/deployment.md)

## Installation

Requires Python 3.10+ and an NVIDIA CUDA GPU with BF16 support. Install from the project directory:

```bash
python -m pip install '.[server]'
```

For the Python API alone, install with `python -m pip install .`. Dependency versions are defined in `pyproject.toml`.

## Start the server

```bash
audiojev-serve --device cuda:0
```

The default model is `shlv/AudioJev`. The first startup downloads the model automatically; subsequent startups reuse the local cache.

The service runs at `http://127.0.0.1:8000`, with interactive API documentation at [/docs](http://127.0.0.1:8000/docs). Check its status:

```bash
curl http://127.0.0.1:8000/health
```

## Send an audio request

Replace `example.wav` with your audio file:

```bash
python examples/http_client.py \
  --audio ./example.wav \
  --question "Which sound is audible?" \
  --options "A dog barking" "A car horn" "Rain falling"
```

The client reads the audio and calls `POST /v1/systemone`, returning the selected answer and each candidate's probability. It uses only the Python standard library. Set `--url` to connect to a different server address.

## Python API

```python
from audiojev_inference import AudioJev, Choice, Noul

model = AudioJev(device="cuda:0")
result = model.system_one(
    state="./example.wav",
    questions={
        "sound": Choice(
            instructions="Which sound is audible?",
            criteria={
                "dog": "A dog barking",
                "horn": "A car horn",
                "rain": "Rain falling",
            },
        ),
        "speech": Noul(instructions="Can speech be heard in this clip?"),
    },
)
print(result["answers"])
```

`state` accepts an audio path, encoded file bytes, or an `AudioInput` waveform. Results are keyed by question ID. Questions in the same request share one audio encoding.

| Type | Input | Output |
|---|---|---|
| `Choice` | A question and candidate descriptions | Selected candidate and probability distribution |
| `Noul` | A statement to evaluate | Probability that the statement is true |
| `Score` | A question and ordered level descriptions | Expected level and probability distribution (experimental) |

See the [API reference](docs/api.md) for candidate keys, probability fields, and input limits.

You can also run the local inference example directly:

```bash
python examples/predict.py \
  --audio ./example.wav \
  --question "Which sound is audible?" \
  --options "A dog barking" "A car horn" "Rain falling"
```

## Deployment and development

See the [deployment guide](docs/deployment.md) for local models, offline operation, GPU configuration, and performance measurement.

Install the development dependencies and run the tests:

```bash
python -m pip install -e '.[server,dev]'
python -m pytest
```

Model use is governed by the [Qwen Research License](https://huggingface.co/shlv/AudioJev/blob/main/LICENSE).
