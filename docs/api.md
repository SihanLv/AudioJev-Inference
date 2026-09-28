# API

## 加载模型

```python
from audiojev_inference import AudioJev

model = AudioJev("shlv/AudioJev", device="cuda:0")
# 本地权重：
model = AudioJev("./models/AudioJev", device="cuda:0", local_files_only=True)
```

模型参数支持 Hugging Face 模型 ID 或本地目录。`revision` 可固定远端提交/标签，`local_files_only=True` 只使用本地目录或已缓存的模型。私有模型使用 `hf auth login` 的凭据或 `HF_TOKEN` 环境变量。

其他加载参数：`max_prompt_tokens=4096`、可选本地 `adapter` 目录及 `merge_adapter=False`。发布的 AudioJev 是完整检查点，正常使用无需 adapter。

## 音频输入

`system_one(state=..., questions=...)` 的 `state` 接受：

| 输入 | 示例 |
|---|---|
| 文件路径 | `"./example.wav"` 或 `Path("./example.wav")` |
| 编码后的音频字节 | `Path("./example.wav").read_bytes()` |
| 波形 | `AudioInput(waveform=mono_array, sample_rate=16000)` |

文件或字节由 SoundFile 解码，支持 WAV、FLAC 等当前 libsndfile 能解码的格式。多声道文件取均值并重采样到 16 kHz。直接提供的波形必须是一维、有限数值数组，且提供正整数采样率。三种音频来源一次只能提供一种。

## 问题类型

### Choice

```python
from audiojev_inference import Choice

question = Choice(
    instructions="Which event is audible?",
    criteria={"dog": "A dog barking", "horn": "A car horn"},
)
```

`criteria` 是 2–36 个非空字符串 key 到非空描述的映射。key 用于结果索引，描述进入模型。返回形状：

```json
{
  "type": "choice",
  "choice": "dog",
  "probabilities": {"dog": 0.8, "horn": 0.2},
  "confidence": 0.2781
}
```

上述数字仅为格式示例。候选概率通过最后一个输入位置上的候选标签 logits 计算，不生成解释文本。

### Noul

```python
from audiojev_inference import Noul

question = Noul(instructions="Can speech be heard in this clip?")
```

返回 `{"type": "noul", "noul": 0.8}`，其中 `noul` 是命题为真的概率。可用 `criteria={"false": "...", "true": "..."}` 自定义真假描述。

### Score

```python
from audiojev_inference import Score

question = Score("Rate speech clarity.", ["Unclear", "Partly clear", "Clear"])
```

`criteria` 是 2–10 个从低到高排列的描述。返回 `score`、`probabilities`、`legend`、`confidence` 和 `experimental: true`。`score` 是从 0 开始的等级索引的概率加权期望；该模型没有针对任意人类评分量表接受监督，使用前需在目标任务上验证。

## 顺序与置信度

Choice/Noul 中，唯一且不含已识别位置指涉的描述会按 UTF-8 字节排序；结果始终绑定回原始 key。若描述重复，或命中 `all/none of the above`、`option A` 等位置指涉规则，则保留输入顺序。这个规则是有限的文本检测，建议使用独立完整的候选描述。Score 保留等级顺序。

`confidence = 1 - H(p)/log(K)`，范围为 0–1，衡量候选分布集中程度。候选顺序稳定性和该数值都不能直接解释为经验正确概率。API 没有隐式的 `other`，可在 Choice 的 `criteria` 中显式提供兜底候选。

## HTTP

`GET /health` 返回已加载模型的状态；服务的交互式文档位于 `/docs`。

`POST /v1/systemone` 示例：

```json
{
  "state": {"audio_base64": "<base64 encoded WAV>"},
  "questions": {
    "sound": {
      "type": "choice",
      "instructions": "Which event is audible?",
      "criteria": {"dog": "A dog barking", "horn": "A car horn"}
    },
    "speech": {"type": "noul", "instructions": "Can speech be heard?"}
  }
}
```

HTTP 请求中的 `state` 只接受 `audio_base64`。`questions` 的每个 value 使用 `type: choice | noul | score`，其他字段与 Python 问题类型一致。可选的 `model` 字段当前固定为 `audiojev-local`。

响应顶层包含 `model`、`answers`、`usage` 和 `confidence_definition`。`answers` 按请求中的问题 ID 索引。`usage` 包含 `input_tokens`、`output_tokens`（此接口为 0）、`audio_encoder_calls` 和 `questions`。

请求格式错误、无效音频、超长输入等返回 HTTP 422。默认最多接收 20,000,000 字节解码后的音频文件；完整输入超过 4,096 个 prompt token 时会拒绝请求。
