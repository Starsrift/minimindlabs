# 07 扩展与综合产出

## 目标

把前面阶段的源码阅读、实验记录和机制理解整合成一个小型扩展任务。重点不是堆功能，而是选择一个明确问题，做出一个可验证产物，并说明它复用了 MiniMind 哪些关键链路。

## 核心文件

按所选方向决定。优先复用前面阶段已经阅读过的源码和文档。

## 阶段主线

```text
选择问题
  -> 定义输入输出
  -> 复用已有阶段理解
  -> 完成一个小产物
  -> 验证结果
  -> 回顾知识链路
```

## 可选方向

- Attention 张量形状追踪脚本
- SFT loss mask 可视化
- MiniMind 模型结构图自动生成
- LoRA 权重合并最小实验
- DPO chosen/rejected 数据样例解释器
- Tool Call 训练样本构造器
- 推理 API 请求与响应示例集
- PPO vs GRPO 对比说明文档

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 选题与边界定义 | `extension-plan.md` |
| 任务 2 | 完成并验证一个最小产物 | 自选产物, `extension-run-log.md` |
| 任务 3 | 回顾产物背后的知识链路 | `knowledge-map.md` |

## 任务清单

### 任务 1：选题与边界定义

- [ ] 从可选方向中选择一个主题。
- [ ] 说明这个主题对应前面哪个阶段。
- [ ] 定义输入、输出、验收标准。
- [ ] 明确不做哪些内容，避免范围过大。

输出：

- `extension-plan.md`

建议记录：

| 项目 | 内容 |
|---|---|
| 选题 |  |
| 对应阶段 |  |
| 输入 |  |
| 输出 |  |
| 验收标准 |  |
| 不做范围 |  |

### 任务 2：完成并验证一个最小产物

- [ ] 如果选脚本，保持输入输出简单清晰。
- [ ] 如果选图表，保证图表能解释关键机制。
- [ ] 如果选文档，保证内容来自前面阶段的真实源码阅读和实验记录。
- [ ] 保留运行命令或生成方式。
- [ ] 运行脚本或检查图表/文档。
- [ ] 记录实际输出、是否符合验收标准、仍存在的问题。

输出：

- 自选产物，例如 `scripts/attention_shape_trace.py`、`figures/model-flow.md`、`notes/ppo-grpo-review.md`。
- `extension-run-log.md`

### 任务 3：回顾产物背后的知识链路

- [ ] 说明扩展产物依赖了哪些阶段知识。
- [ ] 说明这些知识之间如何连接。
- [ ] 画出一个 MiniMind 学习知识图谱。
- [ ] 写出下一步最值得深入的问题。

输出：

- `knowledge-map.md`

建议记录：

```text
model architecture
  -> tokenizer/data
  -> pretrain/SFT
  -> LoRA/DPO/distillation
  -> RLHF/Agent
  -> inference/serving
```

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/07_extensions/
├── tasks.md
├── extension-plan.md
├── extension-run-log.md
├── knowledge-map.md
├── stage-summary.md
└── <自选脚本或文档产物>
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. 选择的扩展方向对应 MiniMind 哪条技术链路？
2. 扩展产物的输入和输出分别是什么？
3. 产物如何验证正确？
4. 它复用了前面哪些阶段的知识？
5. 它暴露出自己当前还不理解哪些问题？
6. 如果继续深入，下一步应该改进什么？

## 验收 Checklist

- [ ] 已完成 `extension-plan.md`。
- [ ] 已明确输入、输出和验收标准。
- [ ] 已完成一个自选脚本、图表或说明文档。
- [ ] 已完成 `extension-run-log.md`。
- [ ] 已完成 `knowledge-map.md`。
- [ ] 已完成 `stage-summary.md`。
