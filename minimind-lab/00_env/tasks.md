# 00 环境与项目入口

## 目标

确认本地能阅读、安装、运行 MiniMind 的最小链路。

## 任务

1. 阅读根目录 `README.md` 的“项目介绍”和“快速开始”。
2. 阅读 `requirements.txt`，标记核心依赖：PyTorch、Transformers、Datasets、FastAPI、Streamlit。
3. 建立 Python 环境，记录安装命令。
4. 确认是否有 GPU；如果没有 GPU，先只做源码阅读和 CPU smoke test。

## 产出

- `env-notes.md`：记录 Python 版本、CUDA/GPU 状态、安装命令、遇到的问题。
- `project-map.md`：用 10 行以内说明每个顶层目录做什么。

## 建议命令

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -r requirements.txt
```
