# 01 模型结构

## 目标

掌握 MiniMind 的 Decoder-only Transformer 实现。

## 核心文件

- `model/model_minimind.py`

## 必读对象

- `MiniMindConfig`
- `RMSNorm`
- `precompute_freqs_cis`
- `Attention`
- `FeedForward`
- `MOEFeedForward`
- `MiniMindBlock`
- `MiniMindForCausalLM`
- `generate`

## 任务

1. 画出 forward 数据流：`input_ids -> embedding -> blocks -> norm -> lm_head -> logits`。
2. 注释 Attention 中 Q/K/V 的张量形状变化。
3. 解释 GQA：`num_attention_heads` 和 `num_key_value_heads` 为什么可以不同。
4. 解释 RoPE 和 YaRN 外推在代码里的位置。
5. 对比 Dense FFN 与 MoE FFN 的差异。

## 产出

- `model-architecture-note.md`
- `attention-shape-trace.md`
- 一张模型结构图，可用 Mermaid。
