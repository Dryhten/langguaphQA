# LangguaphQA

基于 **FastAPI + LangGraph** 的语音问卷对话代理：创建会话、多轮问答并汇总答案。

## 环境要求

- **Python 3.11+**

## 配置

```bash
cd backend
cp .env.example .env
# 编辑 .env：至少填写 OPENAI_LLM_API_KEY（兼容 OpenAI 的密钥）
```

说明见 `backend/.env.example`（兼容 OpenAI 模式地址与模型名可选）。

## 安装依赖

任选其一：

```bash
cd backend
# 使用 uv（推荐，与 pyproject.toml 一致）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

## 启动

在 `backend` 目录下：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

浏览器访问：**http://127.0.0.1:8000/**

根路径由后端直接返回 `frontend/index.html`，API 与页面同源，无需单独配置前端地址。

## 默认问卷 `frontend/survey-default.json`

前端打开时会请求 **`GET /survey-default.json`**，后端从仓库里的 **`frontend/survey-default.json`** 返回 JSON。修改该文件后刷新页面即可更新 textarea 中的默认题目（仍可在页面里临时改后再「开始会话」）。

- **格式**：顶层为数组；每项为对象，至少包含 `id`（字符串）、`text`（题干）、`options`（选项字符串数组），与创建会话时 `POST /sessions` 的 `questions` 一致。
- **注意**：需通过上述后端地址访问页面，`file://` 打开无法拉取该 JSON。若填写了单独的 API 地址，该地址上也需要提供同路径接口（本仓库后端已内置）。

## API 概要

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/sessions` | 创建会话（问卷题目） |
| `POST` | `/sessions/{id}/messages` | 发送用户文本，返回助手回复与当前答案 |

## LangGraph

本项目的对话与问卷状态机在服务端由 **[LangGraph](https://github.com/langchain-ai/langgraph)** 编排：以图（graph）组织节点与边，在多轮用户语音/文本输入下维护会话状态并驱动 LLM 结构化产出。LangGraph 是面向**有状态、可长时间运行**的智能体与工作流的底层编排框架，与 LangChain 生态集成，适合本场景的每轮消息进图、更新「已锁定答案」等流程。仓库与概览见：[langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)；文档见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/)。
