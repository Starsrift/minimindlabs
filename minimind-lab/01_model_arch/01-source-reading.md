# 源码阅读记录：01 模型结构

## 本次任务

- [x] 阅读 `MiniMindConfig`、`MiniMindForCausalLM`、`MiniMindModel`、`MiniMindBlock`。
- [x] 梳理 `input_ids -> embedding -> blocks -> norm -> lm_head -> logits` 主线。
- [x] 说明 `MiniMindModel` 和 `MiniMindForCausalLM` 的分工。
- [x] 说明 labels 和 logits 如何对齐计算 next-token prediction loss。

## 核心文件

- `model/model_minimind.py`

## 核心类和作用

| 对象 | 位置 | 作用 |
|---|---|---|
| `MiniMindConfig` | `model/model_minimind.py` | 定义模型结构超参数，例如 hidden size、层数、词表大小、attention head、RoPE、MoE 等。 |
| `MiniMindBlock` | `model/model_minimind.py` | 单个 Decoder Block，包含 RMSNorm、Self-Attention、FFN/MoE 和残差连接。 |
| `MiniMindModel` | `model/model_minimind.py` | 模型 backbone，负责把 `input_ids` 编码成最后一层 `hidden_states`。 |
| `MiniMindForCausalLM` | `model/model_minimind.py` | Causal LM 任务模型，在 backbone 后接 `lm_head`，输出 `logits` 并计算训练 loss。 |

## 1. MiniMindConfig

`MiniMindConfig` 只保存结构参数，不参与 forward 计算。

重点字段：

| 字段 | 含义 |
|---|---|
| `vocab_size` | 词表大小，决定 embedding 和 `lm_head` 的输出维度。 |
| `hidden_size` | 每个 token 的隐藏向量维度。 |
| `num_hidden_layers` | Transformer Block 的层数。 |
| `num_attention_heads` | Query head 数量。 |
| `num_key_value_heads` | Key/Value head 数量，用于 GQA。 |
| `head_dim` | 每个 attention head 的维度。 |
| `intermediate_size` | Dense FFN 的中间层维度。 |
| `max_position_embeddings` | RoPE 预计算的最大位置长度。 |
| `use_moe` | 是否使用 MoE FFN。 |
| `tie_word_embeddings` | 是否共享输入 embedding 和输出 `lm_head` 权重。 |

理解要点：

```text
MiniMindConfig
  -> 决定 MiniMindModel 如何搭建
  -> 决定 MiniMindBlock 内部 Attention / FFN / MoE 的尺寸
  -> 决定 MiniMindForCausalLM 的 lm_head 输出维度
```

## 2. MiniMindBlock

`MiniMindBlock` 是一个 Decoder-only Transformer Block。

初始化包含：

- `self_attn`：Self-Attention。
- `input_layernorm`：Attention 前的 RMSNorm。
- `post_attention_layernorm`：FFN/MoE 前的 RMSNorm。
- `mlp`：如果 `use_moe=False`，使用 `FeedForward`；否则使用 `MOEFeedForward`。

forward 顺序：

```text
hidden_states
  -> input_layernorm
  -> self_attn
  -> residual add
  -> post_attention_layernorm
  -> FeedForward / MOEFeedForward
  -> residual add
```

关键理解：

- MiniMindBlock 使用 Pre-Norm 结构。
- Attention 前先做 RMSNorm。
- FFN/MoE 前也先做 RMSNorm。
- Attention 和 FFN/MoE 后都做残差连接。

## 3. MiniMindModel

`MiniMindModel` 是模型 backbone，只负责输出 hidden states，不直接输出词表 logits。

初始化包含：

- `embed_tokens`：把 token id 映射到 hidden vector。
- `dropout`：embedding 后 dropout。
- `layers`：堆叠多个 `MiniMindBlock`。
- `norm`：最终 RMSNorm。
- `freqs_cos` / `freqs_sin`：RoPE 位置编码缓存。

forward 主线：

```text
input_ids: [B, S]
  -> embed_tokens
hidden_states: [B, S, H]
  -> dropout
  -> MiniMindBlock x num_hidden_layers
  -> final RMSNorm
hidden_states: [B, S, H]
```

返回值：

```text
hidden_states, presents, aux_loss
```

含义：

- `hidden_states`：最后一层 token 表示。
- `presents`：每层返回的 KV Cache，用于推理复用。
- `aux_loss`：MoE 负载均衡相关辅助损失；Dense FFN 时通常为 0。

## 4. MiniMindForCausalLM

`MiniMindForCausalLM` 是完整的语言模型任务封装。

它包含：

- `self.model = MiniMindModel(config)`
- `self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)`

forward 主线：

```text
input_ids
  -> MiniMindModel
  -> hidden_states: [B, S, H]
  -> lm_head
  -> logits: [B, S, vocab_size]
```

如果传入 `labels`，继续计算 causal LM loss。

## 5. MiniMindModel 和 MiniMindForCausalLM 的分工

| 对象 | 输入 | 输出 | 是否计算 logits | 是否计算 loss |
|---|---|---|---|---|
| `MiniMindModel` | `input_ids` | `hidden_states`, `presents`, `aux_loss` | 否 | 否 |
| `MiniMindForCausalLM` | `input_ids`, `labels` | `logits`, `loss`, `past_key_values`, `hidden_states` | 是 | 是 |

一句话理解：

```text
MiniMindModel 是 backbone，负责理解上下文并输出 hidden states；
MiniMindForCausalLM 是任务头封装，负责把 hidden states 投影成词表 logits，并计算 next-token loss。
```

## 6. labels 和 logits 的对齐方式

源码中的 loss 计算逻辑：

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

假设：

```text
input_ids = [t0, t1, t2, t3]
labels    = [t0, t1, t2, t3]
```

Causal LM 的训练目标是 next-token prediction：

| 使用的 logits | 预测目标 |
|---|---|
| `logits[0]` | `labels[1] = t1` |
| `logits[1]` | `labels[2] = t2` |
| `logits[2]` | `labels[3] = t3` |

所以代码中：

- `logits[..., :-1, :]`：去掉最后一个位置，因为最后一个位置没有下一个 token 可预测。
- `labels[..., 1:]`：去掉第一个 label，因为第一个 token 不是由当前位置预测出来的。
- `ignore_index=-100`：表示该 label 位置不参与 loss。

## 本次阅读结论

MiniMind 的主流程可以概括为：

```text
MiniMindConfig 定义结构参数
  -> MiniMindModel 搭建 backbone
  -> input_ids 经 embedding 和 blocks 得到 hidden_states
  -> MiniMindForCausalLM 用 lm_head 得到 logits
  -> logits 和 labels shift 对齐后计算 next-token prediction loss
```

本任务先建立 forward 主线。Attention 内部 Q/K/V shape、GQA、RoPE、KV Cache、FFN/MoE 细节放到后续任务继续展开。
