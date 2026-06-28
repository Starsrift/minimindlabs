# MiniMind 复现与技术掌握实验室

目标：基于 `jingyaogong/minimind`，系统掌握一个小型 LLM 从模型结构、Tokenizer、数据、预训练、SFT、LoRA、DPO、PPO/GRPO、Agentic RL 到 API 服务的完整技术链路。

## 使用方式

每个阶段都按下面五步推进：

1. 确认当前阶段的所有任务：先阅读本阶段 `tasks.md`，明确目标、核心文件、任务和产出。
2. 阅读源码：定位核心文件、关键类、关键函数和训练/推理入口。
3. 阶段性总结：用自己的话解释公式、张量形状、训练目标、数据流或系统流程。
4. 3060Ti 上复现：优先跑 mini 数据、smoke test 或单卡小步数实验，记录命令、现象和问题。
5. 生成当前阶段的总结性文档：沉淀为 Markdown、命令记录、loss 截图、结构图或可复现实验说明。

建议节奏：每天 1-2 小时，先理解再跑通，不急着追求效果。

每个阶段建议固定产出一个 `stage-summary.md`，写法参考 `PHASE_DOCUMENTATION_WORKFLOW.md`。

## 推荐总路线

| 阶段 | 目录 | 目标 | 核心源码 |
|---|---|---|---|
| 0 | `00_env` | 配环境、理解项目入口 | `README.md`, `requirements.txt` |
| 1 | `01_model_arch` | 掌握 Decoder-only Transformer | `model/model_minimind.py` |
| 2 | `02_tokenizer_data` | 掌握 BPE、chat template、数据 mask | `trainer/train_tokenizer.py`, `dataset/lm_dataset.py` |
| 3 | `03_pretrain_sft` | 跑通 Pretrain 与 SFT | `trainer/train_pretrain.py`, `trainer/train_full_sft.py` |
| 4 | `04_lora_alignment` | 掌握 LoRA、DPO、蒸馏 | `model/model_lora.py`, `trainer/train_dpo.py`, `trainer/train_distillation.py` |
| 5 | `05_rlhf_rlaif_agent` | 掌握 PPO、GRPO、Agentic RL | `trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py` |
| 6 | `06_inference_serving` | 掌握生成、KV Cache、API 服务 | `scripts/serve_openai_api.py`, `scripts/web_demo.py` |
| 7 | `07_extensions` | 做自己的 GitHub Demo 或技术笔记 | 自选 |

## 最终产出

周末至少完成一个可发布成果：

- `MiniMind 技术拆解笔记.md`
- `从零复现 64M LLM 的训练链路.md`
- `MiniMind Tool Use / GRPO 最小实验 Demo`
- `MiniMind 模型结构可视化 + 张量形状注释`

## 验收标准

你真正掌握这个项目时，应该能独立回答：

- MiniMind 和 Qwen/LLaMA 类 Decoder-only 架构有什么对应关系？
- RMSNorm、RoPE、GQA、KV Cache 分别解决什么问题？
- Pretrain、SFT、DPO、PPO、GRPO 的训练目标有什么区别？
- LoRA 为什么能低成本微调？这个仓库的实现简化在哪里？
- Tool Call 是如何通过 chat template、数据格式和 RL 奖励串起来的？
- 这个项目为什么适合教学，但还不是工业级 LLM 系统？
