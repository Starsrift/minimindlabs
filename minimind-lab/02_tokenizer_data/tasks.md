# 02 Tokenizer 与数据

## 目标

理解 MiniMind 如何把原始文本、多轮对话、思考标签和工具调用变成训练样本。

## 核心文件

- `trainer/train_tokenizer.py`
- `model/tokenizer_config.json`
- `dataset/lm_dataset.py`

## 任务

1. 阅读 BPE + ByteLevel tokenizer 的训练流程。
2. 找出特殊 token：`<think>`、`<tool_call>`、`<tool_response>`。
3. 理解 `chat_template` 如何组织 system/user/assistant/tool 消息。
4. 阅读 `SFTDataset.generate_labels`，解释为什么只训练 assistant 部分。
5. 阅读 `DPODataset`，解释 chosen/rejected pair 如何组成 DPO 样本。

## 产出

- `tokenizer-data-note.md`
- `chat-template-examples.md`
- `loss-mask-explanation.md`
