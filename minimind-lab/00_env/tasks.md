# 00 环境与项目入口

## 目标

确认 Mac 本地适合阅读与写笔记，4090 服务器适合运行 MiniMind 的最小链路，并跑通 `minimind-3` dense 权重的最小推理。

今天只做第 0 阶段：环境验证、项目入口阅读、`minimind-3` 权重下载与最小推理。不进入模型结构细读，不开始训练。

## 当前机器分工

- MacBook Pro M1：论文阅读、代码阅读、Markdown 笔记整理。
- RTX 4090 服务器：Python / uv / torch / CUDA 验证，后续训练与推理 smoke test。

## 任务

1. 阅读根目录 `README.md` 的“项目介绍”“快速开始”和 MiniMind3 更新日志。
2. 阅读 `minimind-lab/README.md`，确认整体学习路线与阶段边界。
3. 阅读 `requirements.txt`，标记核心依赖：PyTorch、Transformers、Datasets、FastAPI、Streamlit、ModelScope。
4. 使用 mise 管理语言版本，使用 uv 管理 `.venv` 虚拟环境，并记录 uv pip 镜像源配置。
5. 确认 4090 GPU、CUDA Driver、PyTorch CUDA 版是否可用。
6. 跑一个 CUDA smoke test，确认 torch 可以在 4090 上完成矩阵乘法。
7. 下载 `minimind-3` dense 推理权重。
8. 使用 `eval_llm.py` 跑通一次最小推理，并记录输入、输出和问题。
9. 产出环境记录与项目地图。

## 产出

- `env-notes.md`：记录 Python 版本、CUDA/GPU 状态、安装命令、遇到的问题。
- `project-map.md`：用 10 行以内说明每个顶层目录做什么。

## 建议命令

```bash
cd ~/LLMProjects/Minimind-Labs
source .venv/bin/activate

python --version
uv --version
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

CUDA smoke test:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))

x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print("ok:", y.shape, y.mean().item())
PY
```

检查依赖里是否会覆盖 torch:

```bash
grep -n "torch" requirements.txt
```

uv pip 临时使用镜像源:

```bash
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

常用镜像源:

```text
清华：https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：https://mirrors.aliyun.com/pypi/simple
中科大：https://pypi.mirrors.ustc.edu.cn/simple
腾讯云：https://mirrors.cloud.tencent.com/pypi/simple
华为云：https://repo.huaweicloud.com/repository/pypi/simple
```

项目级固定镜像源可写入 `.mise.toml`:

```toml
[env]
UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

下载 MiniMind3 dense 推理权重:

```bash
uv pip install modelscope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
ls -lh ./minimind-3
```

跑通 MiniMind3 最小推理:

```bash
python eval_llm.py --load_from ./minimind-3
```

## 今日不做

- 不训练模型。
- 不改模型源码。
- 不进入 `01_model_arch` 正式阅读。
- 不下载 `minimind-3-moe`、`minimind-3-gguf`、完整 pretrain / SFT / RLAIF / Agent RL 数据。
- 不直接整包安装依赖覆盖当前可用的 CUDA 版 PyTorch。

## 完成标准

今天结束时能够回答：

1. 4090 是否能被 PyTorch 正常识别？
2. 当前 Python / uv / torch / CUDA 版本分别是什么？
3. MiniMind 顶层目录分别负责什么？
4. 是否已经下载 `minimind-3` 权重？
5. `python eval_llm.py --load_from ./minimind-3` 是否能启动并生成回答？
