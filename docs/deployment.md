# 部署

## 服务进程

从项目根目录安装 `python -m pip install -e '.[server]'` 后：

```bash
audiojev-serve --model shlv/AudioJev --device cuda:0 --host 127.0.0.1 --port 8000
```

模型在进程启动时加载，启动完成后保持驻留。加载使用 BF16、SDPA，关闭语音输出；服务只读取决策位置的候选 logits。

| 参数 | 默认值 / 用途 |
|---|---|
| `--model` / `--model-dir` | `shlv/AudioJev`；也接受本地目录 |
| `--revision` | Hub 分支、标签或完整提交 SHA |
| `--local-files-only` | 只使用已下载的本地模型或缓存 |
| `--device` | `cuda:0` |
| `--host` | `127.0.0.1` |
| `--port` | `8000` |
| `--adapter` | 可选本地 LoRA adapter |
| `--merge-adapter` | 将指定 adapter 合并到内存中的模型 |

`AUDIOJEV_MODEL` 可设置默认模型，兼容旧变量 `AUDIOJEV_MODEL_DIR`。设备和 adapter 可分别通过 `AUDIOJEV_DEVICE`、`AUDIOJEV_ADAPTER` 设置。显式命令行参数优先。

## 下载与离线运行

私有仓库需要有读取权限的账户，可通过 `hf auth login` 登录，或由部署环境注入 `HF_TOKEN`。凭据不应写入项目配置或 Git。

```bash
hf download shlv/AudioJev --local-dir ./models/AudioJev
audiojev-serve --model ./models/AudioJev --local-files-only --device cuda:0
```

直接传 Hub ID 时，服务仅下载权重及 tokenizer/processor 配置到 Hugging Face 缓存。`HF_HOME` 可配置缓存位置；`HF_ENDPOINT` 遵循 Hugging Face 客户端设置。访问私有模型时应使用官方 endpoint，避免镜像无法读取私有仓库的问题。

## 并发与资源

一个进程使用一个 GPU worker，请求在模型内部串行化。同一请求的多个问题共享音频编码，音频缓存只在该请求内有效。多个 GPU 可启动多个独立进程，再由外部代理分配请求。

原始 FP32 权重下载约占 18.8 GB，BF16 参数约占 9.4 GB 显存，运行时还需为音频特征、激活和 CUDA 分配额外显存。实际峰值随输入长度变化。

默认仅监听本机地址。需要对外提供服务时，可在前面配置带认证的反向代理；应用本身没有认证层。外部存活探针可请求 `/health`，但它不执行完整推理。

## 测量音频编码复用

```bash
python examples/benchmark.py --model ./models/AudioJev --audio ./example.wav --questions 4 --repeats 3
```

输出同一音频多问题请求与分开调用的延迟中位数，以及实际音频编码次数。
