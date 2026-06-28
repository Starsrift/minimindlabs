# 03 Pretrain 与 SFT

## 目标

掌握 MiniMind 的两个基础训练阶段：Pretrain 建立续写建模能力，SFT 建立指令跟随能力。这个阶段要把“数据目标差异、loss 计算、训练工程链路、最小实验验证”串成闭环。

## 核心文件

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `dataset/lm_dataset.py`
- `model/model_minimind.py`

## 阶段主线

```text
dataset
  -> dataloader
  -> model forward
  -> shifted logits / labels
  -> CE loss
  -> backward / optimizer
  -> scheduler / checkpoint
  -> smoke test log
```

## 必读对象

- `PretrainDataset`
- `SFTDataset`
- `MiniMindForCausalLM.forward`
- `train_pretrain.py`
- `train_full_sft.py`
- CE loss
- `ignore_index=-100`
- gradient accumulation / clipping
- checkpoint save/resume

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 对比 Pretrain 与 SFT 的训练目标和 labels | `pretrain-vs-sft.md` |
| 任务 2 | 追踪 loss 从 logits 到 backward 的路径 | `training-loop-notes.md` |
| 任务 3 | 梳理训练工程链路：优化器、调度、checkpoint | `training-loop-notes.md` |
| 任务 4 | 跑一次最小训练验证 | `first-run-log.md` |

## 任务清单

### 任务 1：对比 Pretrain 与 SFT 的训练目标和 labels

- [ ] 阅读 `PretrainDataset` 和 `SFTDataset`。
- [ ] 对比两者的输入文本来源、样本结构、labels 构造。
- [ ] 说明 Pretrain 为什么是全序列 next-token prediction。
- [ ] 说明 SFT 为什么只强调 assistant response。
- [ ] 写出一个 Pretrain 样本和一个 SFT 样本的最小示例。

输出：

- `pretrain-vs-sft.md`

建议记录：

| 对比项 | Pretrain | SFT |
|---|---|---|
| 数据形态 | 普通文本 | 多轮指令对话 |
| 训练目标 | 通用 next-token prediction | 学 assistant 回复 |
| labels mask | 通常全序列参与 | 只让目标回复参与 |
| 主要能力 | 语言建模 | 指令跟随 |

### 任务 2：追踪 loss 从 logits 到 backward 的路径

- [ ] 阅读 `MiniMindForCausalLM.forward` 中 loss 的计算。
- [ ] 说明 logits 和 labels 为什么要 shift。
- [ ] 解释 CE loss 的输入 shape。
- [ ] 解释 `ignore_index=-100` 如何影响 loss。
- [ ] 在训练脚本中标出 loss backward 的位置。

输出：

- `training-loop-notes.md` 中的 loss 小节。

### 任务 3：梳理训练工程链路：优化器、调度、checkpoint

- [ ] 阅读 `train_pretrain.py` 和 `train_full_sft.py` 的主训练循环。
- [ ] 记录 forward、loss、backward、grad clip、optimizer step、scheduler step、zero grad 的顺序。
- [ ] 解释梯度累积为什么会影响真实 batch size。
- [ ] 找出 checkpoint 保存位置和保存内容。
- [ ] 说明断点续训需要恢复哪些状态。

输出：

- `training-loop-notes.md` 中的训练循环和 checkpoint 小节。

建议记录：

```text
batch
  -> model forward
  -> loss / accumulation
  -> backward
  -> grad clip
  -> optimizer step
  -> scheduler step
  -> zero grad
  -> save checkpoint
```

### 任务 4：跑一次最小训练验证

- [ ] 选择 Pretrain 或 SFT 的最小数据路径。
- [ ] 使用极小 batch、短序列、少量 step 运行 smoke test。
- [ ] 记录命令、环境、是否跑通、loss 是否有限。
- [ ] 如遇到数据路径、依赖或显存问题，记录问题和处理方式。

输出：

- `first-run-log.md`

命令草稿：

```bash
cd trainer
python train_pretrain.py --epochs 1 --batch_size 2 --num_workers 0 --max_seq_len 128 --device cpu
```

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/03_pretrain_sft/
├── tasks.md
├── pretrain-vs-sft.md
├── training-loop-notes.md
├── first-run-log.md
└── stage-summary.md
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. Pretrain 和 SFT 的训练目标有什么区别？
2. 两者的 labels 构造有什么区别？
3. CE loss、shift logits、shift labels 是如何配合的？
4. `ignore_index=-100` 在训练中解决什么问题？
5. 梯度累积、梯度裁剪、学习率调度分别解决什么问题？
6. checkpoint 保存和断点续训需要哪些状态？
7. 最小训练实验是否跑通，loss 是否正常？

## 验收 Checklist

- [ ] 已完成 `pretrain-vs-sft.md`。
- [ ] 已完成 `training-loop-notes.md`。
- [ ] 已完成 `first-run-log.md`。
- [ ] 已能解释 Pretrain/SFT 训练目标差异。
- [ ] 已能解释 loss 和训练循环。
- [ ] 已完成 `stage-summary.md`。
