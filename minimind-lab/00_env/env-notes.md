# 00 环境记录

## 机器

- 本地机器：MacBook Pro M1
- GPU 服务器：RTX 4090
- 服务器登录：`ssh rtx4090`
- 项目路径：`~/LLMProjects/Minimind-Labs`
- Python 环境管理：mise + uv
- 当前阶段：第 0 阶段，环境与项目入口

## Python / uv

检查命令：

```bash
python --version
uv --version
```

结果：

![image-20260628100504090](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628100504090.png)

结论：

```text
Python 3.12.3 
uv 0.11.25 
```

## GPU / CUDA

检查命令：

```bash
nvidia-smi
```

结果：

![image-20260628100639607](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628100639607.png)

结论：

```text
GPU 型号：4090 RTX
Driver Version ：580.76.05
CUDA Version：13.0
显存：24GB
```

## PyTorch CUDA

检查命令：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

结果：

![image-20260628100906647](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628100906647.png)

```text
2.6.0+cu124
True
NVIDIA GeForce RTX 4090
```

结论：

```text
torch 为 CUDA 版本 --- 2.6.0
RTX 4090 能正常识别
```

## CUDA Smoke Test

检查命令：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print("ok:", y.shape, y.mean().item())
PY
```

结果：

![image-20260628100948381](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628100948381.png)

```text
安装的totch 版本：2.6.0 
安装的cuda：12.4
```

结论：

```text
矩阵乘法成功
GPU 环境通过第 0 阶段验证。
```

## 依赖安装记录

uv pip 镜像源：

```text
推荐：清华源 https://pypi.tuna.tsinghua.edu.cn/simple
备选：阿里云 https://mirrors.aliyun.com/pypi/simple
```

临时使用命令：

```bash
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

项目级配置方式：

```toml
[env]
UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

已安装：

```text
uv、torch、modelscope 或其他依赖。
```

暂不安装 / 暂不覆盖：

```text
避免覆盖当前 CUDA 版 torch。
```

## 权重准备

可选下载命令：

```bash
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
```

当前状态：

![image-20260628101253248](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628101253248.png)

```text
Sucessfully Downloaded
```

目录检查：

```bash
ls -lh ./minimind-3
```

结果：

![image-20260628101335200](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628101335200.png)

```text
./minimind-3 目录就是一个 Transformers/HuggingFace 风格的可推理模型目录。
模型结构配置 + 权重 + tokenizer + 生成配置 + chat 模板 + 说明文档。
```

## minimind-3 权重目录分析

`./minimind-3` 是一个 Transformers 风格的推理模型目录，核心由模型配置、权重、分词器、聊天模板和生成配置组成。

- `config.json`：模型结构配置，定义 hidden size、层数、attention heads、vocab size、RoPE 等参数。
- `model.safetensors`：模型权重文件，保存 embedding、attention、FFN、RMSNorm、lm_head 等训练好的参数。
- `tokenizer.json`：分词器主体，负责文本和 token id 的相互转换。
- `tokenizer_config.json`：分词器使用配置，包含 tokenizer 行为、长度、特殊 token 等信息。
- `special_tokens_map.json`：特殊 token 映射，例如对话起止、结束符、工具调用或思考相关 token。
- `chat_template.jinja`：聊天模板，决定 user/assistant/system 消息如何拼成模型输入。
- `generation_config.json`：生成配置，控制 temperature、top_p、max_new_tokens、eos token 等推理参数。
- `README.md` / `README_en.md`：模型说明文档。
- `minimind.modelfile`：面向 Ollama 或类似工具的模型运行配置。
- `images/`：README 使用的图片资源。

```txt
README.md / README_en.md
模型说明文档。一般包含模型介绍、使用方式、训练信息、示例代码等。第0阶段只需要知道它是“权重包说明书”。
```



```txt
config.json --- 模型结构配置

hidden_size 
num_hidden_layers
num_attention_heads
num_key_value_heads
vocab_size 
max_position_embeddings
rope_theta
architectures
model_type
```



```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "dtype": "float16",
  "head_dim": 96,
  "hidden_act": "silu",
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 2432,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.6",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 6400
}

```



```txt
model.safetensors --- 真正的模型权重文件，约122M，里面是保存的是训练好的参数。
embedding 权重
attention q/k/v/o projection 权重
MLP / FFN 权重
RMSNorm 权重
lm_head 权重
.safetensors 是比传统观念.bin文件更安全、更快加载的权重格式
```



```txt
tokenizer.json --- 分词器主题文件，作用是把文本转成 token id，也把 token id 解码回文本。
EG:
"你好" -> [若干 token ids]
[若干 token ids] -> "你好"
MiniMind3 使用的是 BPE / ByteLevel 风格 tokenizer，并且包含一些特殊标记支持聊天、思考、工具调用。
```



```txt
tokenizer_config.json --- 分词器配置文件
tokenizer 类型
chat_template 路径或内容
bos/eos/pad token 设置
model_max_length
padding/truncation 规则
```

```txt
special_tokens_map.json --- 特殊token映射

bos_token
eos_token
pad_token
unk_token
additional_special_tokens

EG：
<|im_start|>
<|im_end|>
<think>
</think>
<tool_call>
<tool_response>
这些token对chat template、推理停止、工具调用都很重要
```



```txt
chat_template.jinja --- 聊天模版，决定一轮对话如何被拼成模型输入。

EG：
system/user/assistant/messages 
会被格式化成模型训练时见过的文本格式。简单说：同一个问题，如果 chat template 不对，模型效果会明显变差。
```



```txt
generation_config.json --- 生成参数配置，它控制模型回答时的采样方式。比如回答更稳还是更发散，什么时候停止生成。

temperature
top_p
top_k
max_new_tokens
repetition_penalty
eos_token_id
pad_token_id
```



```txt
configuration.json --- 这个文件很小，只有几十字节，可能是 ModelScope/HuggingFace 兼容层或模型仓库元信息，不是核心模型结构配置。
```



```
minimind.modelfile --- 通常用于 Ollama 或类似本地模型运行工具的模型描述文件。
FROM ...
TEMPLATE ...
PARAMETER ...
```



## MiniMind3 最小推理验证

运行命令：

```python
python eval_llm.py --load_from ./minimind-3
```

![image-20260628105056213](/Users/aether/AI Tech/Minimind-Labs/minimind-lab/00_env/assets/image-20260628105056213.png)

测试输入：

```text
💬: 你好，请用一句话介绍你自己。
```

结果：

```text
🧠: 你好！我是一个由jingyaogong创建的高效小参数AI模型，专注于提供精准、快速的文本生成服务。我能够快速处理大量文本信息，提供快速的响应和生成文本支持。
```

结论：

```text
minimind-3 dense 权重已经跑通。
```

## 今日结论

- 4090 环境是否可用：通过
- MiniMind 第 0 阶段是否通过：通过
- 明天是否可以进入 `01_model_arch`：通过
- `minimind-3` 是否跑通：通过
- 遇到的问题：版本软硬件不兼容，原仓库第三方库下载慢
