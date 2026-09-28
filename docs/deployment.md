# Deployment

**English** | [简体中文](deployment.zh-CN.md) · [Home](../README.md)

## Startup and configuration

```bash
audiojev-serve --device cuda:0 --host 127.0.0.1 --port 8000
```

The default model is [shlv/AudioJev](https://huggingface.co/shlv/AudioJev). The first startup downloads it automatically. The HTTP service becomes available after the model has loaded.

| Argument | Default | Purpose |
|---|---|---|
| `--model` | `shlv/AudioJev` | Hugging Face model ID or local directory |
| `--revision` | `main` | Pin a model branch, tag, or full commit SHA |
| `--local-files-only` | Off | Use only local files or an already cached model |
| `--device` | `cuda:0` | Select a GPU |
| `--host` | `127.0.0.1` | Listening address |
| `--port` | `8000` | Listening port |

`--model-dir` is equivalent to `--model`. You can also configure defaults through environment variables:

| Variable | Purpose |
|---|---|
| `AUDIOJEV_MODEL` | Model ID or local directory |
| `AUDIOJEV_DEVICE` | GPU device |
| `HF_HOME` | Hugging Face model cache directory |

Command-line arguments take precedence over environment variables. Run `audiojev-serve --help` for the full argument list.

## Download in advance and run offline

To store the weights in a specific directory:

```bash
hf download shlv/AudioJev --local-dir ./models/AudioJev
audiojev-serve --model ./models/AudioJev --local-files-only --device cuda:0
```

You can also use `--local-files-only` with a model ID to load a previously downloaded version from the Hugging Face cache. Use `--revision` to pin the deployed version.

## Resource requirements

The runtime uses Python 3.10+, PyTorch 2.8, Transformers 4.57.6, and an NVIDIA CUDA GPU with BF16 support.

| Resource | Usage |
|---|---|
| Model files | Approximately 18.8 GB of disk space |
| BF16 model parameters | Approximately 9.4 GB of GPU memory |
| Inference workspace | Additional GPU memory, depending on audio and question length |

The model is loaded at process startup and remains resident in memory. Inference uses BF16 and SDPA.

## Multiple questions and GPUs

A `system_one` request can contain multiple questions. The audio is encoded once, and each question is evaluated separately. Grouping questions about the same audio into one request reduces repeated computation.

Each server process uses one GPU and handles requests serially. For multiple GPUs, start a separate server for each device:

```bash
audiojev-serve --device cuda:0 --port 8000
audiojev-serve --device cuda:1 --port 8001
```

Run these commands in separate terminals or through a process manager. Clients can select a server by port, or a load balancer can distribute requests.

## Service status

`GET /health` returns the service status once the model has loaded. Interactive API documentation is available at `/docs`.

```bash
curl http://127.0.0.1:8000/health
```

## Performance measurement

```bash
python examples/benchmark.py \
  --audio ./example.wav \
  --questions 4 \
  --repeats 3
```

The output compares median latency for a single request containing multiple questions with separate requests for each question, and reports audio encoder call counts. Use `--model` and `--device` to select a model and GPU.
