# Attention Shape Trace：Q/K/V 张量形状追踪

## 1. 本文目标

本文对应 01 阶段任务 2：追踪 MiniMind Attention 中 Q/K/V 的张量形状变化。

重点回答：

```text
输入 hidden_states: [B, S, H]
  -> Q/K/V projection
  -> 多头 view
  -> RoPE
  -> KV Cache
  -> repeat_kv
  -> attention scores
  -> output reshape
  -> 输出: [B, S, H]
```

本任务只关注 Attention 内部 shape，不展开 RoPE 数学推导，也不深入 FFN/MoE。

## 2. 阅读对象

核心文件：

- `model/model_minimind.py`

阅读对象：

- `Attention.__init__`
- `Attention.forward`
- `repeat_kv`
- `apply_rotary_pos_emb`

调用关系：

```text
MiniMindBlock.forward
  -> self.self_attn(...)
  -> Attention.forward(...)
```

也就是说，Attention 是每个 `MiniMindBlock` 内部的第一个核心子层。

## 3. 符号约定

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `S` | 当前输入序列长度，也叫 `seq_len` |
| `S_past` | KV Cache 中历史 token 长度 |
| `S_total` | 当前 K/V 总长度，`S_past + S` |
| `H` | hidden size |
| `V` | vocab size，本任务中暂不使用 |
| `n_heads` | Query head 数量，也就是 `num_attention_heads` |
| `n_kv_heads` | Key/Value head 数量，也就是 `num_key_value_heads` |
| `D` | head dim，也就是 `head_dim` |
| `n_rep` | 每个 KV head 被复制给多少个 Query head 使用 |

核心关系：

```text
H = n_heads * D
n_rep = n_heads // n_kv_heads
```

如果使用 GQA，通常：

```text
n_heads > n_kv_heads
```

## 4. Attention 初始化参数

`Attention.__init__` 中最重要的结构参数是：

| 参数 | 含义 |
|---|---|
| `n_local_heads` | Query head 数量 |
| `n_local_kv_heads` | Key/Value head 数量 |
| `n_rep` | K/V head 重复倍数 |
| `head_dim` | 每个 head 的维度 |
| `q_proj` | 从 hidden states 投影出 Query |
| `k_proj` | 从 hidden states 投影出 Key |
| `v_proj` | 从 hidden states 投影出 Value |
| `o_proj` | 把多头 attention 输出投影回 hidden size |
| `q_norm` | 对每个 Query head 做 RMSNorm |
| `k_norm` | 对每个 Key head 做 RMSNorm |

投影层维度：

| 层 | 输入维度 | 输出维度 |
|---|---|---|
| `q_proj` | `H` | `n_heads * D` |
| `k_proj` | `H` | `n_kv_heads * D` |
| `v_proj` | `H` | `n_kv_heads * D` |
| `o_proj` | `n_heads * D` | `H` |

关键点：

```text
Q 使用 n_heads。
K/V 使用 n_kv_heads。
当 n_heads != n_kv_heads 时，就是 GQA 结构。
```

## 5. Attention.forward 的代码顺序

Attention 内部顺序可以概括为：

```text
x: [B, S, H]
  -> q_proj / k_proj / v_proj
  -> view 成多头
  -> q_norm / k_norm
  -> RoPE 作用于 Q/K
  -> 拼接 past_key_value
  -> 保存新的 past_kv
  -> repeat_kv 对齐 K/V head 数
  -> transpose 到 attention 计算格式
  -> scaled dot-product attention
  -> transpose + reshape 回 [B, S, H]
  -> o_proj
```

注意两个顺序：

```text
RoPE 在 KV Cache 拼接之前已经作用于当前 Q/K。
repeat_kv 发生在 KV Cache 拼接之后。
```

## 6. 无 KV Cache 时的 shape 追踪

### 6.1 输入

Attention 的输入来自 `MiniMindBlock.forward` 中的 `input_layernorm(hidden_states)`。

shape 不变：

```text
x: [B, S, H]
```

### 6.2 Q/K/V projection

代码逻辑：

```text
xq = q_proj(x)
xk = k_proj(x)
xv = v_proj(x)
```

shape：

```text
Q: [B, S, n_heads * D]
K: [B, S, n_kv_heads * D]
V: [B, S, n_kv_heads * D]
```

### 6.3 view 成多头

projection 后会把最后一维拆成 head 数和 head dim：

```text
Q: [B, S, n_heads, D]
K: [B, S, n_kv_heads, D]
V: [B, S, n_kv_heads, D]
```

这一步的意义：

```text
把一个大 hidden vector 拆成多个 attention head。
```

### 6.4 q_norm / k_norm

MiniMind 对 Q/K 的每个 head 做 RMSNorm：

```text
Q: [B, S, n_heads, D]
K: [B, S, n_kv_heads, D]
```

shape 不变。

### 6.5 RoPE

RoPE 只作用在 Q/K 上，不作用在 V 上。

```text
Q: [B, S, n_heads, D]
K: [B, S, n_kv_heads, D]
V: [B, S, n_kv_heads, D]
```

RoPE 改变的是 Q/K 向量里的数值，不改变 shape。

### 6.6 repeat_kv

如果 `n_heads > n_kv_heads`，K/V 需要通过 `repeat_kv` 对齐到 Query head 数。

```text
K: [B, S, n_kv_heads, D] -> [B, S, n_heads, D]
V: [B, S, n_kv_heads, D] -> [B, S, n_heads, D]
```

如果 `n_rep=1`，说明 Q head 数和 K/V head 数相同，不需要重复。

关键理解：

```text
repeat_kv 不是重新计算 K/V，而是复用已有的 K/V head。
```

### 6.7 transpose 到 attention 计算格式

Attention 计算需要把 head 维度放到前面：

```text
Q: [B, S, n_heads, D] -> [B, n_heads, S, D]
K: [B, S, n_heads, D] -> [B, n_heads, S, D]
V: [B, S, n_heads, D] -> [B, n_heads, S, D]
```

### 6.8 attention scores

注意力分数来自：

```text
scores = Q @ K^T / sqrt(D)
```

shape：

```text
Q:      [B, n_heads, S, D]
K^T:    [B, n_heads, D, S]
scores: [B, n_heads, S, S]
```

因为是 causal attention，当前位置不能看未来位置，所以会加上 causal mask。

### 6.9 attention output

softmax 后的 scores 再乘 V：

```text
scores: [B, n_heads, S, S]
V:      [B, n_heads, S, D]
output: [B, n_heads, S, D]
```

然后 transpose + reshape：

```text
[B, n_heads, S, D]
  -> [B, S, n_heads, D]
  -> [B, S, n_heads * D]
  -> [B, S, H]
```

最后经过 `o_proj`：

```text
[B, S, H] -> [B, S, H]
```

所以 Attention 子层的输入输出 shape 是一致的：

```text
输入: [B, S, H]
输出: [B, S, H]
```

## 7. 有 KV Cache 时的 shape 变化

KV Cache 用于推理阶段。

假设当前只输入新 token 或一小段新 token：

```text
当前序列长度: S
历史缓存长度: S_past
总 K/V 长度: S_total = S_past + S
```

当前 K/V 在 RoPE 后 shape 是：

```text
当前 K: [B, S, n_kv_heads, D]
当前 V: [B, S, n_kv_heads, D]
```

历史缓存：

```text
past K: [B, S_past, n_kv_heads, D]
past V: [B, S_past, n_kv_heads, D]
```

拼接后：

```text
K: [B, S_total, n_kv_heads, D]
V: [B, S_total, n_kv_heads, D]
```

然后再 repeat_kv：

```text
K: [B, S_total, n_heads, D]
V: [B, S_total, n_heads, D]
```

transpose 后：

```text
Q: [B, n_heads, S, D]
K: [B, n_heads, S_total, D]
V: [B, n_heads, S_total, D]
```

scores shape：

```text
scores: [B, n_heads, S, S_total]
```

这表示当前 token 可以 attend 到历史 token 和当前 token 范围内允许看到的位置。

## 8. 具体配置示例

使用一个具体配置：

```text
hidden_size = 768
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 96
n_rep = 2
B = 2
S = 16
```

验证关系：

```text
H = n_heads * D = 8 * 96 = 768
n_rep = n_heads // n_kv_heads = 8 // 4 = 2
```

无 KV Cache 时：

| 步骤 | shape |
|---|---|
| 输入 `x` | `[2, 16, 768]` |
| `q_proj(x)` | `[2, 16, 768]` |
| `k_proj(x)` | `[2, 16, 384]` |
| `v_proj(x)` | `[2, 16, 384]` |
| Q view | `[2, 16, 8, 96]` |
| K view | `[2, 16, 4, 96]` |
| V view | `[2, 16, 4, 96]` |
| RoPE 后 Q/K | shape 不变 |
| repeat K | `[2, 16, 8, 96]` |
| repeat V | `[2, 16, 8, 96]` |
| Q transpose | `[2, 8, 16, 96]` |
| K transpose | `[2, 8, 16, 96]` |
| V transpose | `[2, 8, 16, 96]` |
| attention scores | `[2, 8, 16, 16]` |
| attention output | `[2, 8, 16, 96]` |
| reshape | `[2, 16, 768]` |
| `o_proj` 后 | `[2, 16, 768]` |

## 9. 完整 shape 表

| 步骤 | Q shape | K shape | V shape | 说明 |
|---|---|---|---|---|
| 输入 | `[B,S,H]` | `[B,S,H]` | `[B,S,H]` | block 输入 |
| projection | `[B,S,n_heads*D]` | `[B,S,n_kv_heads*D]` | `[B,S,n_kv_heads*D]` | 线性投影 |
| view | `[B,S,n_heads,D]` | `[B,S,n_kv_heads,D]` | `[B,S,n_kv_heads,D]` | 拆成多头 |
| q/k norm | `[B,S,n_heads,D]` | `[B,S,n_kv_heads,D]` | 不变 | Q/K 做 RMSNorm |
| RoPE | `[B,S,n_heads,D]` | `[B,S,n_kv_heads,D]` | 不变 | RoPE 只作用 Q/K |
| KV Cache | 不变 | `[B,S_total,n_kv_heads,D]` | `[B,S_total,n_kv_heads,D]` | 推理时拼接历史 K/V |
| repeat_kv | 不需要 | `[B,S_total,n_heads,D]` | `[B,S_total,n_heads,D]` | GQA 对齐 Q head 数 |
| transpose | `[B,n_heads,S,D]` | `[B,n_heads,S_total,D]` | `[B,n_heads,S_total,D]` | attention 计算格式 |
| scores | - | - | - | `[B,n_heads,S,S_total]` |
| attention output | - | - | - | `[B,n_heads,S,D]` |
| reshape | - | - | - | `[B,S,H]` |
| `o_proj` | - | - | - | `[B,S,H]` |

无 KV Cache 时：

```text
S_total = S
```

有 KV Cache 时：

```text
S_total = S_past + S
```

## 10. GQA 与 repeat_kv

GQA 是 Grouped Query Attention。

在 MiniMind 中，Q head 数和 K/V head 数可以不同：

```text
Q:   n_heads
K/V: n_kv_heads
```

如果：

```text
n_heads = 8
n_kv_heads = 4
```

那么：

```text
n_rep = 8 // 4 = 2
```

含义是：每个 K/V head 会被两个 Query head 共享。

`repeat_kv` 的作用：

```text
[B, S, n_kv_heads, D]
  -> [B, S, n_kv_heads, n_rep, D]
  -> [B, S, n_heads, D]
```

直观理解：

```text
Q 头更多，用来保留更强的查询表达能力；
K/V 头更少，用来减少 KV Cache 和计算/显存开销。
```

## 11. Flash Attention 与手写 Attention 分支

MiniMind 中有两个 attention 计算分支：

```text
1. Flash / scaled_dot_product_attention 分支
2. 手写 scores + softmax + V 分支
```

二者的输入 shape 一致：

```text
Q: [B, n_heads, S, D]
K: [B, n_heads, S_total, D]
V: [B, n_heads, S_total, D]
```

二者的输出 shape 也一致：

```text
output: [B, n_heads, S, D]
```

区别主要在实现方式和性能上，不改变本任务关注的 shape 结论。

## 12. 本任务结论

MiniMind Attention 的输入和输出 shape 都是：

```text
[B, S, H]
```

内部 shape 变化可以概括为：

```text
[B, S, H]
  -> Q/K/V projection
  -> [B, S, heads, D]
  -> RoPE on Q/K
  -> concat KV Cache
  -> repeat_kv for GQA
  -> [B, heads, S, D]
  -> attention scores
  -> [B, S, H]
```

最关键的三个理解：

1. Q 使用 `num_attention_heads`，K/V 使用 `num_key_value_heads`。
2. RoPE 只改变 Q/K 的数值，不改变 shape，也不作用于 V。
3. `repeat_kv` 是为了让 K/V head 数对齐 Q head 数，它复用 K/V，不重新计算 K/V。
