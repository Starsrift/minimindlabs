# 02 Tokenizer 与数据

## 目标

掌握 MiniMind 如何把原始文本、多轮对话、思考标签和工具调用转换为可训练样本。这个阶段的重点不是单独背 tokenizer 或 dataset API，而是把“文本如何变成 input_ids，哪些 token 参与 loss，偏好样本如何组织”这条链路讲清楚。

## 核心文件

- `trainer/train_tokenizer.py`
- `model/tokenizer_config.json`
- `dataset/lm_dataset.py`

## 阶段主线

```text
raw text / conversation
  -> tokenizer
  -> special tokens
  -> chat_template
  -> input_ids
  -> labels / loss mask
  -> Pretrain / SFT / DPO samples
```

## 必读对象

- BPE tokenizer 训练流程
- ByteLevel pre-tokenizer
- `tokenizer_config.json`
- special tokens
- `chat_template`
- `PretrainDataset`
- `SFTDataset`
- `SFTDataset.generate_labels`
- `DPODataset`

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 建立 tokenizer 与特殊 token 基础认知 | `tokenizer-data-note.md` |
| 任务 2 | 还原 chat template 到 input_ids 的转换链路 | `chat-template-examples.md` |
| 任务 3 | 对比 Pretrain 与 SFT 的 labels 构造 | `loss-mask-explanation.md` |
| 任务 4 | 理解 DPO 偏好样本如何复用对话编码链路 | `tokenizer-data-note.md` |

## 任务清单

### 任务 1：建立 tokenizer 与特殊 token 基础认知

- [ ] 阅读 `trainer/train_tokenizer.py`，说明 tokenizer 的训练输入、词表大小、保存位置。
- [ ] 解释 BPE 和 ByteLevel 在文本切分中的分工。
- [ ] 阅读 `model/tokenizer_config.json`，记录 bos/eos/pad/unk 等基础 token。
- [ ] 找出 `<think>`、`<tool_call>`、`<tool_response>` 等和 reasoning/tool use 相关的特殊 token。

输出：

- `tokenizer-data-note.md` 中的 tokenizer 训练与特殊 token 小节。

### 任务 2：还原 chat template 到 input_ids 的转换链路

- [ ] 阅读 `chat_template`。
- [ ] 构造一个普通 system/user/assistant 多轮对话样例。
- [ ] 构造一个包含 `<think>` 的样例。
- [ ] 构造一个包含 `<tool_call>` 和 `<tool_response>` 的样例。
- [ ] 对每个样例写出 `messages -> rendered text -> input_ids` 的转换说明。

输出：

- `chat-template-examples.md`

建议记录：

```text
messages
  -> chat_template 渲染为单段文本
  -> tokenizer 编码为 input_ids
  -> dataset 构造 labels
```

### 任务 3：对比 Pretrain 与 SFT 的 labels 构造

- [ ] 阅读 `PretrainDataset`，说明普通文本如何构造成 `input_ids` 和 `labels`。
- [ ] 阅读 `SFTDataset` 和 `SFTDataset.generate_labels`。
- [ ] 解释为什么 SFT 只让 assistant 回复部分参与 loss。
- [ ] 说明 `ignore_index=-100` 如何屏蔽 system/user/tool 等非目标 token。
- [ ] 画出一个具体样例中的 `input_ids`、`labels`、loss mask 对齐关系。

输出：

- `loss-mask-explanation.md`

建议记录：

| token 片段 | 属于谁 | input_ids | labels | 是否计入 loss |
|---|---|---|---|---|
| system prompt | system | 有 | `-100` | 否 |
| user message | user | 有 | `-100` | 否 |
| assistant answer | assistant | 有 | token id | 是 |

### 任务 4：理解 DPO 偏好样本如何复用对话编码链路

- [ ] 阅读 `DPODataset`。
- [ ] 说明 prompt、chosen answer、rejected answer 的数据结构。
- [ ] 解释 chosen/rejected 是否共享同一个 prompt。
- [ ] 对比 DPO 样本和 SFT 样本在编码、mask、训练目标上的差异。
- [ ] 写出一个最小 chosen/rejected pair 示例。

输出：

- `tokenizer-data-note.md` 中的 Pretrain/SFT/DPO 数据对比小节。

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/02_tokenizer_data/
├── tasks.md
├── tokenizer-data-note.md
├── chat-template-examples.md
├── loss-mask-explanation.md
└── stage-summary.md
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. MiniMind 的 tokenizer 是如何训练和保存的？
2. BPE 和 ByteLevel 分别解决什么问题？
3. special tokens 在 thinking 和 tool use 中有什么作用？
4. `chat_template` 如何把多轮消息渲染成模型输入？
5. Pretrain 和 SFT 的 labels 构造有什么差异？
6. 为什么 SFT 只训练 assistant 部分？
7. DPO 的 chosen/rejected pair 如何复用对话编码链路？

## 验收 Checklist

- [ ] 已完成 `tokenizer-data-note.md`。
- [ ] 已完成 `chat-template-examples.md`。
- [ ] 已完成 `loss-mask-explanation.md`。
- [ ] 已能解释 tokenizer 训练、special tokens 和 chat template。
- [ ] 已能解释 Pretrain/SFT/DPO 的样本构造差异。
- [ ] 已完成 `stage-summary.md`。
