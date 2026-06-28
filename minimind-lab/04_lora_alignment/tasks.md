# 04 LoRA、DPO 与蒸馏

## 目标

掌握 MiniMind 中三类对齐和迁移方法：LoRA 解决低成本参数微调，DPO 解决偏好对齐，蒸馏解决能力迁移或模型压缩。本阶段重点是区分它们的训练对象、数据格式、loss 目标和工程路径。

## 核心文件

- `model/model_lora.py`
- `trainer/train_lora.py`
- `trainer/train_dpo.py`
- `trainer/train_distillation.py`

## 阶段主线

```text
base model
  -> parameter-efficient tuning / preference optimization / distillation
  -> task-specific data
  -> task-specific loss
  -> adapted student / policy model
```

## 必读对象

- LoRA A/B 矩阵
- LoRA 注入范围
- `train_lora.py`
- `dpo_loss`
- `train_dpo.py`
- `distillation_loss`
- `train_distillation.py`

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 理解 LoRA 如何低成本改造模型 | `lora-note.md` |
| 任务 2 | 理解 DPO 如何用偏好对更新 policy | `dpo-note.md` |
| 任务 3 | 理解蒸馏如何让 student 学 teacher | `distillation-note.md` |
| 任务 4 | 横向比较 SFT、LoRA、DPO、蒸馏 | `alignment-method-comparison.md` |

## 任务清单

### 任务 1：理解 LoRA 如何低成本改造模型

- [ ] 阅读 `model/model_lora.py`。
- [ ] 找出 LoRA A/B 矩阵的定义和 forward 叠加方式。
- [ ] 解释为什么 B 矩阵通常初始化为 0。
- [ ] 标记仓库中 LoRA 注入的 `nn.Linear` 范围。
- [ ] 阅读 `train_lora.py`，记录模型加载、参数冻结、训练参数筛选和权重保存流程。

输出：

- `lora-note.md`

建议记录：

```text
base Linear
  -> add low-rank A/B branch
  -> freeze base weights
  -> train LoRA weights
  -> save adapter
```

### 任务 2：理解 DPO 如何用偏好对更新 policy

- [ ] 阅读 `train_dpo.py`。
- [ ] 阅读 `dpo_loss`。
- [ ] 说明 prompt、chosen、rejected 的数据结构。
- [ ] 说明 policy model 和 reference model 的作用。
- [ ] 解释 DPO loss 如何鼓励 chosen 优于 rejected。

输出：

- `dpo-note.md`

### 任务 3：理解蒸馏如何让 student 学 teacher

- [ ] 阅读 `train_distillation.py`。
- [ ] 阅读 `distillation_loss`。
- [ ] 说明 teacher model 和 student model 的分工。
- [ ] 解释 KL loss、temperature、hard label loss 的含义。
- [ ] 记录蒸馏训练的数据输入、teacher 输出、student 输出和最终 loss。

输出：

- `distillation-note.md`

### 任务 4：横向比较 SFT、LoRA、DPO、蒸馏

- [ ] 对比四者的数据格式。
- [ ] 对比四者训练哪些参数。
- [ ] 对比四者的 loss 目标。
- [ ] 对比四者适合解决的问题。
- [ ] 说明它们在完整 LLM 训练链路中的前后关系。

输出：

- `alignment-method-comparison.md`

建议记录：

| 方法 | 数据格式 | 训练目标 | 训练参数 | 解决问题 |
|---|---|---|---|---|
| SFT | prompt/answer | 学标准回答 | 全量或部分 | 指令跟随 |
| LoRA | prompt/answer | 低成本适配 | LoRA 参数 | 省显存微调 |
| DPO | prompt/chosen/rejected | 偏好 chosen | policy | 偏好对齐 |
| 蒸馏 | input + teacher logits | 逼近 teacher | student | 压缩/迁移能力 |

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/04_lora_alignment/
├── tasks.md
├── lora-note.md
├── dpo-note.md
├── distillation-note.md
├── alignment-method-comparison.md
└── stage-summary.md
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. LoRA A/B 矩阵如何接入原始 Linear？
2. 为什么 LoRA 可以减少可训练参数？
3. DPO 的 chosen/rejected pair 如何进入 loss？
4. reference model 在 DPO 中有什么作用？
5. 蒸馏中的 KL loss 和 temperature 分别解决什么问题？
6. SFT、LoRA、DPO、蒸馏分别适合什么训练目标？

## 验收 Checklist

- [ ] 已完成 `lora-note.md`。
- [ ] 已完成 `dpo-note.md`。
- [ ] 已完成 `distillation-note.md`。
- [ ] 已完成 `alignment-method-comparison.md`。
- [ ] 已能区分 LoRA、DPO、蒸馏的目标和数据。
- [ ] 已完成 `stage-summary.md`。
