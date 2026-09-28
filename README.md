# AudioJev-Inference

AudioJev 的独立 Python 推理库与 HTTP 服务。输入音频和问题，返回候选概率、二元判断或等级评分；同一音频的多个问题共享一次音频编码。

默认加载 Hugging Face 上的 **[shlv/AudioJev](https://huggingface.co/shlv/AudioJev)**（RD-SKL，λ=0.5，seed=20261001）。模型权重由 Hugging Face 管理，本项目包含服务代码、客户端示例和测试，可单独克隆、安装和部署。

## 安装

需要 Python 3.10+ 和支持 BF16 的 NVIDIA CUDA GPU。以下命令均在 **AudioJev-Inference 仓库根目录**执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[server]"
```

项目使用 PyTorch 2.8、Transformers 4.57.6 和 SDPA。请使用与本机 CUDA 环境匹配的 PyTorch 构建。访问私有模型时，先登录有读取权限的 Hugging Face 账户：

```bash
export HF_ENDPOINT=https://huggingface.co
hf auth login
```

服务会自动下载并缓存模型。也可以先下载到本地：

```bash
hf download shlv/AudioJev --local-dir ./models/AudioJev
```

`models/` 已加入 `.gitignore`。原始 FP32 权重约 18.8 GB，加载时转换为 BF16。

## 启动服务

```bash
audiojev-serve --model shlv/AudioJev --device cuda:0
```

默认地址为 `http://127.0.0.1:8000`。模型加载完成后，可以检查状态或打开交互式 API 文档：

```bash
curl http://127.0.0.1:8000/health
# 浏览器打开 http://127.0.0.1:8000/docs
```

从本地模型目录离线启动：

```bash
audiojev-serve --model ./models/AudioJev --local-files-only --device cuda:0
```

`--model-dir` 是 `--model` 的兼容别名。也可通过 `AUDIOJEV_MODEL` 或 `AUDIOJEV_MODEL_DIR` 指定模型，使用 `--revision` 固定 Hugging Face 提交或标签。更多设置见 [部署说明](docs/deployment.md)。

## 调用 HTTP 接口

服务提供 `POST /v1/systemone`，接收 base64 音频和问题。附带客户端可以直接读取音频文件发送请求：

```bash
python examples/http_client.py --audio ./example.wav --question "Which sound is audible?" --options "A dog barking" "A car horn" "Rain falling"
```

客户端仅使用 Python 标准库，不加载模型，也不需要 GPU。请求格式和响应字段见 [API 文档](docs/api.md)。

## Python API

```python
from audiojev_inference import AudioJev, Choice, Noul

model = AudioJev("shlv/AudioJev", device="cuda:0")
result = model.system_one(
    state="./example.wav",
    questions={
        "sound": Choice(
            instructions="Which sound is audible?",
            criteria={"dog": "A dog barking", "horn": "A car horn", "rain": "Rain falling"},
        ),
        "speech": Noul(instructions="Can speech be heard in this clip?"),
    },
)
print(result["answers"])
print(result["usage"]["audio_encoder_calls"])  # 同一请求的多个问题共享一次音频编码
```

`AudioJev()` 默认使用 `shlv/AudioJev`；传入本地目录可加载已下载的权重。调用方可以传文件路径、音频字节或 `AudioInput` 波形。音频统一转换为 16 kHz 单声道。

命令行单次推理示例：

```bash
python examples/predict.py --audio ./example.wav --question "Which sound is audible?" --options "A dog barking" "A car horn" "Rain falling"
```

## 概率与候选顺序

- `Choice` 支持 2–36 个候选，返回候选 key、完整概率分布和 `confidence`。
- `Noul` 返回命题为真的概率。
- `Score` 支持 2–10 个有序等级，返回期望等级与分布；该接口目前为实验性功能。
- `confidence = 1 - H(p)/log(K)` 表示分布集中程度，不是预测正确率。
- `Choice` 和 `Noul` 对唯一、无位置指涉的描述按 UTF-8 字节排序，并把结果映射回原始 key；重复或位置指涉描述保留输入顺序。`Score` 始终保留等级顺序。这是服务的序列化约定，与模型卡中原始候选顺序的评测设置不同。

所有概率只在给定候选上归一化。需要兜底答案时，在候选列表中显式提供。接口细节见 [API 文档](docs/api.md)。

## 项目结构

```text
AudioJev-Inference/
├── audiojev_inference/   # Python API、模型加载与 HTTP 服务
├── examples/            # 本地推理、HTTP 客户端、性能测量
├── docs/                # API 与部署文档
├── tests/               # 无 GPU 的单元测试
├── pyproject.toml       # 安装依赖与 audiojev-serve 命令
└── README.md
```

## 开发与测量

```bash
python -m pip install -e ".[server,dev]"
python -m pytest
python examples/benchmark.py --audio ./example.wav --questions 4
```

单元测试不下载权重，也不需要 GPU；性能测量需要模型和 GPU。

模型的使用条件见 [AudioJev 模型卡](https://huggingface.co/shlv/AudioJev)及其 Qwen Research License。项目采用 `state`、`questions`、`answers` 的类型化接口，计算在本地完成。
