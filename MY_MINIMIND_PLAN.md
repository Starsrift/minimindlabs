# My MiniMind 学习与创新计划

这个分支是我基于 MiniMind 做大模型系统学习、复现和二次创新的个人工作区。

## 分支定位

- 原始项目：`jingyaogong/minimind`
- 本地学习分支：`study/minimind-lab`
- 个人目标：掌握小型 LLM 从模型结构到训练、对齐、Agentic RL、推理服务的完整链路。

## 工作原则

1. 先复现，再改造。
2. 先写清楚技术笔记，再做功能创新。
3. 每个实验都保留命令、配置、现象和结论。
4. 不直接破坏原仓库主线代码；新增内容优先放在 `minimind-lab/`。
5. 形成可以发布到 GitHub 的学习笔记、实验脚本或小 Demo。

## 当前学习入口

从这里开始：

- `minimind-lab/README.md`
- `minimind-lab/00_env/tasks.md`
- `minimind-lab/01_model_arch/tasks.md`

## Fork 后推荐远端设置

在 GitHub 上 fork 原仓库后，可以把你的 fork 加成新 remote：

```bash
git remote rename origin upstream
git remote add origin git@github.com:<your-name>/minimind.git
git push -u origin study/minimind-lab
```

之后：

- `upstream` 用来同步原作者更新。
- `origin` 用来推送你的学习分支和创新代码。

## 第一阶段目标

完成以下 3 个产出：

1. `minimind-lab/01_model_arch/model-architecture-note.md`
2. `minimind-lab/02_tokenizer_data/tokenizer-data-note.md`
3. `minimind-lab/03_pretrain_sft/pretrain-vs-sft.md`

## 创新方向候选

- MiniMind 模型结构中文注释版
- Attention 张量形状追踪脚本
- SFT loss mask 可视化工具
- LoRA / DPO 最小可运行实验
- Tool Call 数据构造器
- PPO、GRPO、DPO 对齐算法对比笔记
- 面向北邮计算机学生的大模型系统复现教程
