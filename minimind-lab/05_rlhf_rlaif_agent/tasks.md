# 05 RLHF / RLAIF / Agentic RL

## 目标

理解 MiniMind 中强化学习对齐的教学实现。这个阶段不要求把 PPO/GRPO 当成孤立公式背下来，而是要围绕“模型生成回答、环境或规则给 reward、算法更新 policy”这条闭环，理解 PPO、GRPO 和 Agentic RL 的差异。

## 核心文件

- `trainer/train_ppo.py`
- `trainer/train_grpo.py`
- `trainer/train_agent.py`
- `trainer/rollout_engine.py`

## 阶段主线

```text
prompt
  -> policy rollout
  -> reward scoring
  -> PPO / GRPO objective
  -> policy update
  -> tool-use rollout
  -> behavior alignment
```

## 必读对象

- rollout 生成流程
- reward 计算
- PPO actor/critic/reference model
- PPO advantage / KL penalty
- GRPO 多回答采样与组内标准化
- Agentic RL 工具定义与执行
- `rollout_engine.py`

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 先建立 rollout 与 reward 的公共闭环 | `reward-design-review.md` |
| 任务 2 | 对比 PPO 与 GRPO 的优化方式 | `ppo-grpo-comparison.md` |
| 任务 3 | 理解 Agentic RL 的工具调用训练闭环 | `agentic-rl-note.md` |
| 任务 4 | 判断教学实现与真实 RLHF/RLAIF 系统差距 | `rlhf-system-gap.md` |

## 任务清单

### 任务 1：先建立 rollout 与 reward 的公共闭环

- [ ] 阅读 rollout 相关代码，说明 prompt 如何进入 policy 生成。
- [ ] 找出 reward 计算入口。
- [ ] 记录格式分、长度分、重复惩罚、结果正确性等 reward 组成。
- [ ] 说明不同 reward 项分别约束什么行为。
- [ ] 判断 reward 是否可能被模型投机利用。

输出：

- `reward-design-review.md`

建议记录：

```text
prompt
  -> policy generate
  -> parse response
  -> compute reward
  -> return rollout data for optimization
```

### 任务 2：对比 PPO 与 GRPO 的优化方式

- [ ] 阅读 `train_ppo.py`，说明 actor、critic、reward model、reference model 的分工。
- [ ] 解释 advantage 和 KL penalty 在 PPO 中的作用。
- [ ] 阅读 `train_grpo.py`，说明同 prompt 多回答如何组成一组。
- [ ] 解释组内 reward 标准化如何构造相对优势。
- [ ] 对比 PPO 和 GRPO 的状态、成本、稳定性和实现复杂度。

输出：

- `ppo-grpo-comparison.md`

建议记录：

| 对比项 | PPO | GRPO |
|---|---|---|
| 是否需要 critic | 需要 | 通常不需要 |
| 优势估计 | advantage | 组内相对 reward |
| 样本组织 | rollout batch | 同 prompt 多回答 |
| 主要约束 | KL penalty | 相对偏好和 KL |

### 任务 3：理解 Agentic RL 的工具调用训练闭环

- [ ] 阅读 `train_agent.py`。
- [ ] 阅读 `rollout_engine.py`。
- [ ] 找出工具定义、工具调用格式和解析逻辑。
- [ ] 说明模型如何生成 tool call。
- [ ] 说明工具执行结果如何作为 tool response 回填到上下文。
- [ ] 说明 reward 如何评价工具调用格式和结果正确性。

输出：

- `agentic-rl-note.md`

建议记录：

```text
prompt
  -> model generates tool_call
  -> parse tool_call
  -> execute / simulate tool
  -> append tool_response
  -> continue generation
  -> compute reward
```

### 任务 4：判断教学实现与真实 RLHF/RLAIF 系统差距

- [ ] 判断当前实现是否包含真实人工偏好数据。
- [ ] 判断 reward model 是否足够独立和可靠。
- [ ] 判断工具执行是否接近真实环境。
- [ ] 判断 rollout、评测、安全约束与真实系统的差距。
- [ ] 写出这个阶段继续深入时最值得补的能力。

输出：

- `rlhf-system-gap.md`

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/05_rlhf_rlaif_agent/
├── tasks.md
├── reward-design-review.md
├── ppo-grpo-comparison.md
├── agentic-rl-note.md
├── rlhf-system-gap.md
└── stage-summary.md
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. rollout 和 reward 在 RLHF/RLAIF 中分别负责什么？
2. PPO 中 actor、critic、reward model、reference model 分别负责什么？
3. KL penalty 为什么是 RLHF 中的重要约束？
4. GRPO 如何用组内相对 reward 替代 critic？
5. Agentic RL 中 tool call 如何生成、解析、执行和回填？
6. 这个仓库的教学实现和真实 RLHF/RLAIF 系统有哪些差距？

## 验收 Checklist

- [ ] 已完成 `reward-design-review.md`。
- [ ] 已完成 `ppo-grpo-comparison.md`。
- [ ] 已完成 `agentic-rl-note.md`。
- [ ] 已完成 `rlhf-system-gap.md`。
- [ ] 已能解释 rollout、reward、PPO、GRPO、Agentic RL 的关系。
- [ ] 已完成 `stage-summary.md`。
