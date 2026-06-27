# 05 RLHF / RLAIF / Agentic RL

## 目标

理解 PPO、GRPO、CISPO 以及多轮 Tool Use 的训练闭环。

## 核心文件

- `trainer/train_ppo.py`
- `trainer/train_grpo.py`
- `trainer/train_agent.py`
- `trainer/rollout_engine.py`

## 任务

1. 阅读 PPO：actor、critic、reward model、advantage、KL penalty。
2. 阅读 GRPO：多生成样本、组内 reward 标准化、无 critic 的相对优化。
3. 阅读 Agentic RL：工具定义、工具调用解析、模拟执行、reward 设计。
4. 总结这个仓库的 reward 是如何混合格式分、长度分、重复惩罚和结果正确性的。
5. 判断教学实现和真实 RLHF 系统的差距。

## 产出

- `ppo-grpo-comparison.md`
- `agentic-rl-note.md`
- `reward-design-review.md`
