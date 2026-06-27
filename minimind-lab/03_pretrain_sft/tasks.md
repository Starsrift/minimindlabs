# 03 Pretrain 与 SFT

## 目标

掌握语言模型训练的两个基础阶段：续写能力与指令跟随能力。

## 核心文件

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`

## 任务

1. 对比 PretrainDataset 和 SFTDataset 的 label 构造差异。
2. 理解 CE loss、ignore_index=-100、梯度累积、梯度裁剪。
3. 记录训练脚本中的学习率调度、保存 checkpoint、断点续训逻辑。
4. 尝试用 mini 数据跑一个极小步数实验。

## 产出

- `pretrain-vs-sft.md`
- `training-loop-notes.md`
- `first-run-log.md`

## 最小实验命令草稿

```bash
cd trainer
python train_pretrain.py --epochs 1 --batch_size 2 --num_workers 0 --max_seq_len 128 --device cpu
```

CPU 只适合 smoke test；正式训练建议上 GPU。
