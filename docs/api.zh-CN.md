# API

[English](api.md) | **简体中文** · [首页](../README.zh-CN.md)

## 加载模型

```python
from audiojev_inference import AudioJev

model = AudioJev(device="cuda:0")
```

默认加载 [shlv/AudioJev](https://huggingface.co/shlv/AudioJev)，自动下载并缓存所需文件。也可以指定模型 ID 或本地目录：

```python
model = AudioJev("./models/AudioJev", device="cuda:0", local_files_only=True)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model_dir` | `shlv/AudioJev` | 模型 ID 或本地目录 |
| `device` | `cuda:0` | 使用的 CUDA 设备 |
| `revision` | `None` | 模型分支、标签或提交 SHA；未指定时使用主分支 |
| `local_files_only` | `False` | 仅加载本地文件或已有缓存 |
| `max_prompt_tokens` | `4096` | 完整输入的 token 上限 |

模型实例可重复使用，适合在程序启动时创建一次。

## 音频输入

`model.system_one(state=..., questions=...)` 接收一段音频和一个问题映射。

| `state` 类型 | 示例 |
|---|---|
| 文件路径 | `"./example.wav"` 或 `Path("./example.wav")` |
| 编码后的音频字节 | `Path("./example.wav").read_bytes()` |
| 波形 | `AudioInput(waveform=mono_array, sample_rate=16000)` |

```python
from audiojev_inference import AudioInput

audio = AudioInput(waveform=mono_array, sample_rate=16000)
```

文件和字节支持 SoundFile 能解码的格式，例如 WAV、FLAC。多声道文件自动混为单声道，并重采样到 16 kHz。直接提供的波形须为一维数组，采样率为正整数。一次只提供一种音频来源。

## Choice：候选选择

```python
from audiojev_inference import Choice

question = Choice(
    instructions="Which event is audible?",
    criteria={"dog": "A dog barking", "horn": "A car horn"},
)
```

`criteria` 是 2–36 个候选 key 到描述的映射。key 用于索引输出，描述用于模型判断。二者均须为非空字符串。

返回示例：

```json
{
  "type": "choice",
  "choice": "dog",
  "probabilities": {"dog": 0.8, "horn": 0.2},
  "confidence": 0.2781
}
```

`choice` 是所选候选的 key，`probabilities` 包含所有候选的概率。上述数值用于说明返回格式。

## Noul：真假判断

```python
from audiojev_inference import Noul

question = Noul(instructions="Can speech be heard in this clip?")
```

返回形如 `{"type": "noul", "noul": 0.8}`，其中 `noul` 为命题成立的概率。

可通过 `criteria={"false": "...", "true": "..."}` 自定义真假描述。

## Score：等级评分

```python
from audiojev_inference import Score

question = Score(
    instructions="Rate speech clarity.",
    criteria=["Unclear", "Partly clear", "Clear"],
)
```

`criteria` 是按等级排列的 2–10 个描述。返回 `score`、`probabilities`、`legend`、`confidence` 和 `experimental: true`。

等级索引从 0 开始，`score` 是其概率加权期望。例如三个等级的概率为 `[0.1, 0.2, 0.7]` 时，评分为 `1.6`。Score 是实验性接口，模型未针对任意人工评分量表接受专门训练。

## 概率与候选描述

概率在提供的候选之间归一化。需要“其他”答案时，可将其作为一个显式候选加入 `criteria`。

`confidence = 1 - H(p)/log(K)` 衡量分布集中程度，范围为 0–1；它不是预测正确率。

Choice/Noul 会对唯一且不含已识别位置指涉的描述按 UTF-8 字节排序，结果始终映射回原始 key。重复描述或包含 `all/none of the above`、`option A` 等位置指涉时保留输入顺序。使用独立、完整的候选描述最便于表达问题。Score 始终保留等级顺序。

## 多问题响应

`questions` 的 key 是调用方指定的问题 ID。`system_one` 按这些 ID 返回答案：

```python
result = model.system_one(
    state="./example.wav",
    questions={"event": question},
)
answer = result["answers"]["event"]
```

| 顶层字段 | 含义 |
|---|---|
| `model` | 服务模型标识 `audiojev-local` |
| `answers` | 问题 ID 到答案的映射 |
| `usage` | 输入 token 数、问题数和音频编码次数 |
| `confidence_definition` | `confidence` 字段的定义 |

`usage` 包含 `input_tokens`、`output_tokens`、`audio_encoder_calls` 和 `questions`。此接口直接计算候选概率，`output_tokens` 为 0；同一请求的多个问题共享一次音频编码。

## HTTP 接口

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

HTTP 请求的 `state` 使用 `audio_base64`。问题对象通过 `type: choice | noul | score` 指定类型，其他字段与对应 Python 类型一致。响应结构与 `system_one` 相同。

请求可省略 `model`；提供时其值为 `audiojev-local`。

### GET /health

模型加载完成后返回：

```json
{"status": "ok", "model": "audiojev-local"}
```

完整交互式文档位于 `/docs`。

### 输入限制

- base64 解码后的音频文件大小上限为 20,000,000 字节。
- 音频、问题和候选组成的完整 prompt 上限为 4,096 个 token。
- 音频须包含至少 25 ms 的有效采样。

请求格式错误、无效音频或超出输入限制时，HTTP 接口返回 422，并在 `detail` 中说明原因。
