# 06 推理与服务

## 目标

掌握 MiniMind 如何生成文本、暴露 OpenAI API、支持 thinking 和 tool_calls。

## 核心文件

- `model/model_minimind.py`
- `scripts/serve_openai_api.py`
- `scripts/chat_api.py`
- `scripts/web_demo.py`
- `scripts/convert_model.py`

## 任务

1. 阅读 `generate`，理解 temperature、top_p、top_k、repetition_penalty。
2. 理解 KV Cache 在推理中的作用。
3. 阅读 `serve_openai_api.py`，解释 `/v1/chat/completions` 如何兼容 OpenAI 风格。
4. 阅读 `parse_response`，理解 reasoning_content 和 tool_calls 的解析。
5. 尝试启动本地 API 或 WebUI。

## 产出

- `generation-note.md`
- `openai-api-serving-note.md`
- `tool-call-serving-note.md`
