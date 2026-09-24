# AudioJev inference

独立可安装的本地推理包。加载 AudioJev 训练后的 Qwen2.5-Omni 3B 全参数
checkpoint，或基座加 LoRA adapter；Python 和 HTTP 接口采用 Jev 的
`state`、`questions`、`answers` 结构。只接收音频，不调用 TypeSafe 服务。

## 安装

在具备 CUDA 的环境中安装与训练一致的 PyTorch 和依赖：

在 `inference/` 目录执行：

```bash
python -m pip install -e '.[server]'
```

本机已安装训练环境时可使用：

```bash
cd inference
/data/lsh/conda_envs/audiojev/bin/python -m pip install -e . --no-deps
```

全参数模型目录必须包含 `config.json`、Safetensors 权重和 processor 文件。
本机按 development loss 选出的 seed 20261001 模型为：

```text
/data/lsh/AudioJev/runs/full_parameter_v1/full_seed20261001/general/checkpoint-2048/model
```

也可将 `model_dir` 设为原始 `/data/lsh/models/Qwen2.5-Omni-3B`，并传入
`adapter` 目录。权重不随本包分发。

## Python

```python
from audiojev_inference import AudioJev, Choice, Noul, Score

jev = AudioJev("/path/to/trained/checkpoint/model", device="cuda:0")
result = jev.system_one(
    state="/path/to/audio.wav",
    questions={
        "intent": Choice(
            instructions="说话者的主要请求是什么？",
            criteria={"refund": "申请退款", "delivery": "查询配送状态", "other": "其他请求"},
        ),
        "laughter": Noul(instructions="片段中是否可以听到笑声？"),
        "quality": Score(
            instructions="评估人声清晰程度。",
            criteria=["难以辨认", "部分清晰", "清晰"],
        ),
    },
)
print(result["answers"])
```

`state` 也接受音频文件字节或 `AudioInput(waveform=mono_array,
sample_rate=16000)`。音频会转换为 16 kHz 单声道。一个请求的所有问题共用
一次解码和一次音频塔编码。问题 ID 和候选 key 仅用于结果索引；模型读取
`instructions` 与候选描述，所以描述应包含完整语义。

`choice` 返回所选 key、完整分布与 `confidence`；`noul` 返回真值概率；
`score` 返回等级期望、等级分布和 legend。`confidence` 明确定义为
`1 - H(p)/log(K)`，描述分布集中程度，**不是**正确概率，也不等同于
TypeSafe 的内部计算。Score 目前没有人类量表监督，输出带
`"experimental": true`；请先做任务内验证再使用。

Choice/Noul 最多 36 个候选，Score 最多 10 级。只对提供的候选做 softmax，
没有隐式 `other`；需要兜底时应显式写入 criteria。当前接口支持字符串
描述；TypeSafe 接口中的对象/数组描述尚未实现。

## HTTP

```bash
audiojev-serve --model-dir /path/to/trained/checkpoint/model --device cuda:0
```

`POST /v1/systemone` 接收 JSON。可选 `model` 字段为 `audiojev-local`。
`state.audio_base64` 是编码后音频文件
的 Base64；服务默认只监听 `127.0.0.1`，限 20 MB 音频。示例：

```json
{
  "state": {"audio_base64": "<base64 encoded WAV>"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "说话者的主要请求是什么？",
      "criteria": {"refund": "申请退款", "delivery": "查询配送状态"}
    },
    "laughter": {"type": "noul", "instructions": "片段中是否可以听到笑声？"}
  }
}
```

`GET /health` 只确认模型已加载。服务使用一个 GPU worker；请求在模型
内部串行化，保证请求级音频缓存不会串音。如需并发吞吐，可为每张 GPU
启动一个独立进程，由外层负载均衡。

## 加速与测量

- 模型常驻显存，BF16 + SDPA，关闭音频输出。
- 直接读取最终决策位置 logits，不调用逐 token `generate`，也不计算整段
  prompt 的词表投影。
- 单次多问题请求复用音频塔输出；文本前向仍随问题数增长。
- 可选 `merge_adapter=True` 把 LoRA 合并到内存中的基座，需先核对合并前后
  的决策是否符合使用要求；全参数 checkpoint 无需合并。

`usage.audio_encoder_calls` 可核对实际音频塔调用次数。测量多问题收益：

```bash
python benchmark.py --model-dir /path/to/model --audio /path/to/audio.wav --questions 4
```

参考接口：[Choice](https://docs.typesafe.ai/primitives/choice)、
[Noul](https://docs.typesafe.ai/primitives/noul)、
[Score](https://docs.typesafe.ai/primitives/score)。本包复用请求/响应形状，
不声称概率或性能与 Jev 相同。
