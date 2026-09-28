# AudioJev-Inference

[English](README.md) | **简体中文**

AudioJev 的 Python 推理库与 HTTP 服务。输入一段音频和自然语言问题，即可获得候选答案的概率、真假判断或等级评分。同一音频可以一次回答多个问题。

模型：[shlv/AudioJev](https://huggingface.co/shlv/AudioJev) · [API 文档](docs/api.zh-CN.md) · [部署说明](docs/deployment.zh-CN.md)

## 安装

需要 Python 3.10+ 和支持 BF16 的 NVIDIA CUDA GPU。在项目目录安装：

```bash
python -m pip install '.[server]'
```

仅使用 Python API 时，安装 `python -m pip install .` 即可。依赖版本由 `pyproject.toml` 管理。

## 启动服务

```bash
audiojev-serve --device cuda:0
```

默认使用 `shlv/AudioJev`。首次启动会自动下载模型，后续启动复用本地缓存。

服务地址为 `http://127.0.0.1:8000`，交互式 API 文档位于 [/docs](http://127.0.0.1:8000/docs)。检查服务状态：

```bash
curl http://127.0.0.1:8000/health
```

## 发送音频请求

将 `example.wav` 替换为你的音频文件：

```bash
python examples/http_client.py \
  --audio ./example.wav \
  --question "Which sound is audible?" \
  --options "A dog barking" "A car horn" "Rain falling"
```

客户端读取音频并调用 `POST /v1/systemone`，返回所选答案及各候选的概率。客户端只需 Python 标准库，可通过 `--url` 指定服务地址。

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

`state` 接受音频路径、文件字节或 `AudioInput` 波形。结果按问题 ID 返回；同一请求中的多个问题共享音频编码。

| 类型 | 输入 | 输出 |
|---|---|---|
| `Choice` | 问题与候选描述 | 所选候选、概率分布 |
| `Noul` | 待判断的命题 | 命题为真的概率 |
| `Score` | 问题与有序等级描述 | 期望等级、概率分布（实验性接口） |

候选 key、概率字段和输入限制见 [API 文档](docs/api.zh-CN.md)。

也可以直接运行本地推理示例：

```bash
python examples/predict.py \
  --audio ./example.wav \
  --question "Which sound is audible?" \
  --options "A dog barking" "A car horn" "Rain falling"
```

## 部署与开发

本地模型、离线运行、GPU 配置和性能测量见 [部署说明](docs/deployment.zh-CN.md)。

开发环境安装与测试：

```bash
python -m pip install -e '.[server,dev]'
python -m pytest
```

模型使用条款见 [Qwen Research License](https://huggingface.co/shlv/AudioJev/blob/main/LICENSE)。
