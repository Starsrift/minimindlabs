# MiniMind 项目地图

- `README.md`：原项目介绍、MiniMind3 更新日志、快速开始、训练与推理说明。
- `requirements.txt`：项目 Python 依赖列表，安装前需要确认是否会覆盖 CUDA 版 PyTorch。
- `model/`：MiniMind 模型结构、LoRA 结构与 tokenizer 配置。
- `trainer/`：Tokenizer、Pretrain、SFT、DPO、PPO、GRPO、蒸馏、Agentic RL 等训练入口。
- `dataset/`：训练数据读取、样本构造与数据说明。
- `scripts/`：模型转换、Web Demo、OpenAI API 服务、工具调用评测与聊天脚本。
- `eval_llm.py`：本地加载模型权重并进行推理验证的入口。
- `images/`：项目说明图、演示图与 README 资源。
- `minimind-lab/`：个人阶段化学习、复现记录、代码阅读笔记与 GitHub 发布材料。
- `MY_MINIMIND_PLAN.md`：个人 MiniMind 学习与创新计划。
