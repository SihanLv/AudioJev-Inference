# 部署

## 启动与配置

```bash
audiojev-serve --device cuda:0 --host 127.0.0.1 --port 8000
```

默认模型为 [shlv/AudioJev](https://huggingface.co/shlv/AudioJev)。首次启动自动下载，加载完成后提供 HTTP 服务。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--model` | `shlv/AudioJev` | Hugging Face 模型 ID 或本地目录 |
| `--revision` | `main` | 固定模型分支、标签或完整提交 SHA |
| `--local-files-only` | 关闭 | 仅使用本地文件或已缓存的模型 |
| `--device` | `cuda:0` | 指定 GPU |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8000` | 监听端口 |

`--model-dir` 与 `--model` 等价。也可通过环境变量配置默认值：

| 变量 | 用途 |
|---|---|
| `AUDIOJEV_MODEL` | 模型 ID 或本地目录 |
| `AUDIOJEV_DEVICE` | GPU 设备 |
| `HF_HOME` | Hugging Face 模型缓存目录 |

命令行参数优先于环境变量。完整参数可通过 `audiojev-serve --help` 查看。

## 预先下载与离线运行

需要将权重放在指定目录时：

```bash
hf download shlv/AudioJev --local-dir ./models/AudioJev
audiojev-serve --model ./models/AudioJev --local-files-only --device cuda:0
```

`--local-files-only` 也可以配合模型 ID 使用，此时从 Hugging Face 缓存加载已经下载的版本。使用 `--revision` 可固定部署版本。

## 资源需求

运行环境为 Python 3.10+、PyTorch 2.8、Transformers 4.57.6，以及支持 BF16 的 NVIDIA CUDA GPU。

| 资源 | 用量 |
|---|---|
| 模型文件 | 约 18.8 GB 磁盘空间 |
| BF16 模型参数 | 约 9.4 GB 显存 |
| 推理工作空间 | 额外显存，随音频和问题长度变化 |

模型在进程启动时加载并保持驻留。推理使用 BF16 和 SDPA。

## 多问题与多 GPU

一次 `system_one` 请求可以包含多个问题。音频只编码一次，各问题分别进行决策；将同一音频的问题放入一个请求可减少重复计算。

每个服务进程使用一个 GPU，请求在进程内串行执行。多 GPU 部署可为各设备启动独立服务：

```bash
audiojev-serve --device cuda:0 --port 8000
audiojev-serve --device cuda:1 --port 8001
```

上述命令分别运行于不同终端或进程管理器中，客户端通过端口选择服务，也可由负载均衡器分配请求。

## 服务状态

`GET /health` 在模型加载完成后返回状态；`/docs` 提供交互式接口文档。

```bash
curl http://127.0.0.1:8000/health
```

## 性能测量

```bash
python examples/benchmark.py \
  --audio ./example.wav \
  --questions 4 \
  --repeats 3
```

输出同一音频多问题请求与逐问题调用的延迟中位数，以及音频编码次数。可使用 `--model` 和 `--device` 指定模型与 GPU。
