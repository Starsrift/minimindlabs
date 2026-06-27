# 04 LoRA、DPO 与蒸馏

## 目标

掌握低成本微调、偏好对齐和知识蒸馏。

## 核心文件

- `model/model_lora.py`
- `trainer/train_lora.py`
- `trainer/train_dpo.py`
- `trainer/train_distillation.py`

## 任务

1. 阅读 LoRA 的 A/B 矩阵实现，解释为什么 B 初始化为 0。
2. 标记这个仓库 LoRA 注入范围：只给部分 `nn.Linear` 增加 LoRA。
3. 阅读 `dpo_loss`，写出 DPO 的核心公式含义。
4. 阅读 `distillation_loss`，解释 KL loss、temperature、teacher/student。
5. 比较 SFT、DPO、蒸馏各自需要的数据格式。

## 产出

- `lora-note.md`
- `dpo-note.md`
- `distillation-note.md`
