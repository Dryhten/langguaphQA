from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(Path(__file__).resolve().parent / ".env")

from models import AgentTurnOutput


def _compat_base_url() -> str | None:
    url = (os.getenv("OPENAI_LLM_BASE_URL") or "").strip()
    return url or None


def _llm_api_key() -> str:
    key = (os.getenv("OPENAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "请在 .env 中设置 OPENAI_LLM_API_KEY（兼容模式也可使用 OPENAI_API_KEY）。"
        )
    return key


def _llm_model_name() -> str:
    name = (os.getenv("OPENAI_LLM_MODEL") or "").strip()
    if name:
        return name
    return (os.getenv("QWEN_MODEL") or "qwen-plus").strip() or "qwen-plus"


def get_chat_llm() -> BaseChatModel:
    """若设置 OPENAI_LLM_BASE_URL，走 DashScope OpenAI 兼容接口；否则走原生 ChatTongyi。"""
    base = _compat_base_url()
    if base:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=base.rstrip("/"),
            api_key=_llm_api_key(),
            model=_llm_model_name(),
            temperature=0.2,
        )

    from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=(os.getenv("QWEN_MODEL") or "qwen-plus").strip() or "qwen-plus",
        temperature=0.2,
        dashscope_api_key=_llm_api_key(),
    )


def get_chat_tongyi() -> BaseChatModel:
    """与 get_chat_llm 相同，保留别名供旧代码引用。"""
    return get_chat_llm()


def get_structured_agent_llm():
    """优先 function_calling；兼容接口上通常比 DashScope 原生 structured 更稳。"""
    llm = get_chat_llm()
    try:
        return llm.with_structured_output(AgentTurnOutput, method="function_calling")
    except (TypeError, ValueError):
        pass
    return llm.with_structured_output(AgentTurnOutput)
