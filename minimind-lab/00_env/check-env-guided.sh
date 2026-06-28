#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_header() {
  printf "\n============================================================\n"
  printf "%s\n" "$1"
  printf "============================================================\n"
}

ask_run() {
  local title="$1"
  local command="$2"

  printf "\n下一步将检测：%s\n" "$title"
  printf "将执行命令：%s\n" "$command"
  read -r -p "请输入 y 执行，n 跳过 [y/n]: " answer

  case "$answer" in
    y|Y|yes|YES)
      printf "\n>>> %s\n" "$command"
      bash -lc "$command"
      local status=$?
      if [ "$status" -eq 0 ]; then
        printf "\n[OK] %s 检测完成。\n" "$title"
      else
        printf "\n[WARN] %s 检测命令返回非 0 状态：%s\n" "$title" "$status"
      fi
      ;;
    n|N|no|NO)
      printf "[SKIP] 已跳过：%s\n" "$title"
      ;;
    *)
      printf "[SKIP] 输入不是 y/n，默认跳过：%s\n" "$title"
      ;;
  esac
}

print_header "MiniMind 第 0 阶段：交互式环境检查"

printf "项目根目录：%s\n" "$ROOT_DIR"
printf "建议在 4090 服务器的项目根目录运行：\n"
printf "  cd ~/LLMProjects/Minimind-Labs\n"
printf "  source .venv/bin/activate\n"
printf "  bash minimind-lab/00_env/check-env-guided.sh\n"

ask_run "当前目录与项目文件" "pwd && ls -lh"
ask_run "Python 版本" "python --version"
ask_run "uv 版本" "uv --version"
ask_run "mise 版本" "mise --version"
ask_run "虚拟环境路径" "python -c 'import sys; print(sys.executable); print(sys.prefix)'"
ask_run "GPU 与 CUDA Driver 状态 nvidia-smi" "nvidia-smi"
ask_run "nvcc 编译器版本" "nvcc --version"
ask_run "PyTorch CUDA 可用性" "python -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')\""
ask_run "CUDA 矩阵乘法 smoke test" "python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available')
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('ok:', y.shape, float(y.mean()))
PY"
ask_run "requirements.txt 是否包含 torch" "grep -n \"torch\" requirements.txt || true"
ask_run "ModelScope 命令是否可用" "modelscope --version"
ask_run "minimind-3 权重目录主要文件" "ls -lh ./minimind-3"
ask_run "minimind-3 config.json 关键信息" "python - <<'PY'
import json
from pathlib import Path
p = Path('minimind-3/config.json')
if not p.exists():
    raise SystemExit('minimind-3/config.json not found')
cfg = json.loads(p.read_text())
keys = [
    'architectures',
    'model_type',
    'vocab_size',
    'hidden_size',
    'num_hidden_layers',
    'num_attention_heads',
    'num_key_value_heads',
    'max_position_embeddings',
    'rope_theta',
]
for k in keys:
    print(f'{k}: {cfg.get(k)}')
PY"
ask_run "MiniMind3 最小推理入口" "python eval_llm.py --load_from ./minimind-3"

print_header "检查结束"
printf "请把关键输出补充到：minimind-lab/00_env/env-notes.md\n"
printf "如果某一步失败，优先记录：命令、报错、你的判断。\n"
