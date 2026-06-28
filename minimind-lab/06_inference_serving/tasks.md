# 06 推理与服务

## 目标

掌握 MiniMind 从模型推理到服务接口的完整链路。这个阶段要把 `generate`、KV Cache、采样参数、OpenAI 风格 API、thinking/tool_calls 解析和本地运行验证串起来。

## 核心文件

- `model/model_minimind.py`
- `scripts/serve_openai_api.py`
- `scripts/chat_api.py`
- `scripts/web_demo.py`
- `scripts/convert_model.py`

## 阶段主线

```text
messages / prompt
  -> tokenizer / chat_template
  -> generate
  -> sampling + KV Cache
  -> parse response
  -> OpenAI-style API / WebUI
```

## 必读对象

- `generate`
- temperature / top_p / top_k / repetition_penalty
- `past_key_values`
- `serve_openai_api.py`
- `/v1/chat/completions`
- `parse_response`
- `reasoning_content`
- `tool_calls`
- `web_demo.py`

## 任务总览

| 任务 | 学习重点 | 输出文档 |
|---|---|---|
| 任务 1 | 理解单机生成内核：generate、采样、KV Cache | `generation-note.md` |
| 任务 2 | 理解服务层如何兼容 OpenAI Chat Completions | `openai-api-serving-note.md` |
| 任务 3 | 理解 reasoning_content 与 tool_calls 的结构化解析 | `tool-call-serving-note.md` |
| 任务 4 | 启动本地 API 或 WebUI 做端到端验证 | `serving-run-log.md` |

## 任务清单

### 任务 1：理解单机生成内核：generate、采样、KV Cache

- [ ] 阅读 `model/model_minimind.py` 中的 `generate`。
- [ ] 说明生成循环中每一步输入和输出。
- [ ] 解释 temperature、top_p、top_k、repetition_penalty 分别影响什么。
- [ ] 说明 `past_key_values` 如何传入和返回。
- [ ] 对比 prefill 和 decode 阶段的计算差异。

输出：

- `generation-note.md`

建议记录：

```text
prefill: full prompt -> build K/V cache
decode: last token -> reuse past K/V -> append new K/V
```

### 任务 2：理解服务层如何兼容 OpenAI Chat Completions

- [ ] 阅读 `scripts/serve_openai_api.py`。
- [ ] 找出 `/v1/chat/completions` 路由。
- [ ] 说明请求中的 model、messages、temperature、stream 等字段如何处理。
- [ ] 说明返回结构如何兼容 OpenAI 风格。
- [ ] 对比普通响应和流式响应。

输出：

- `openai-api-serving-note.md`

### 任务 3：理解 reasoning_content 与 tool_calls 的结构化解析

- [ ] 阅读 `parse_response`。
- [ ] 说明 `reasoning_content` 如何从模型输出中解析。
- [ ] 说明 tool_calls 如何从文本解析为结构化字段。
- [ ] 构造一个包含 thinking 的输出示例。
- [ ] 构造一个包含 tool call 的输出示例。
- [ ] 说明解析失败时可能出现什么问题。

输出：

- `tool-call-serving-note.md`

### 任务 4：启动本地 API 或 WebUI 做端到端验证

- [ ] 选择 API 或 WebUI 作为验证入口。
- [ ] 记录启动命令。
- [ ] 发送一个普通对话请求。
- [ ] 发送一个 thinking 或 tool call 相关请求。
- [ ] 记录请求、响应、报错和解决方式。

输出：

- `serving-run-log.md`

## 阶段输出文件

完成后，本阶段目录建议包含：

```text
minimind-lab/06_inference_serving/
├── tasks.md
├── generation-note.md
├── openai-api-serving-note.md
├── tool-call-serving-note.md
├── serving-run-log.md
└── stage-summary.md
```

## 阶段完成标准

完成本阶段后，需要能独立回答：

1. `generate` 的主循环如何工作？
2. temperature、top_p、top_k、repetition_penalty 分别影响什么？
3. KV Cache 在 prefill 和 decode 阶段分别做什么？
4. `/v1/chat/completions` 如何兼容 OpenAI 风格？
5. `reasoning_content` 如何解析？
6. tool_calls 如何从模型输出解析为结构化响应？
7. 本地 API 或 WebUI 是否能启动并返回结果？

## 验收 Checklist

- [ ] 已完成 `generation-note.md`。
- [ ] 已完成 `openai-api-serving-note.md`。
- [ ] 已完成 `tool-call-serving-note.md`。
- [ ] 已完成 `serving-run-log.md`。
- [ ] 已能解释模型推理、服务接口和响应解析之间的关系。
- [ ] 已完成 `stage-summary.md`。
