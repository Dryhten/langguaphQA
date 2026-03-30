from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from llm import get_chat_llm, get_structured_agent_llm
from models import AgentTurnOutput
from state import AgentState


def _format_questionnaire(questionnaire: list[dict[str, Any]], answers: dict[str, str]) -> str:
    lines: list[str] = []
    for q in questionnaire:
        qid = q["id"]
        status = "已锁定" if qid in answers else "未完成"
        cur = answers.get(qid, "")
        opts = " | ".join(q["options"])
        lines.append(
            f"- [{status}] id={qid}\n  题干：{q['text']}\n  选项（须完全一致）：{opts}\n  当前选择：{cur or '（无）'}"
        )
    return "\n".join(lines)


def build_system_prompt(questionnaire: list[dict[str, Any]], answers: dict[str, str]) -> str:
    pending = [q["id"] for q in questionnaire if q["id"] not in answers]
    done = {qid: answers[qid] for qid in answers}
    return f"""你是一个友善的**持续语音**问卷助手：用户会一直用语音补充信息，你要**反复理解整段对话**，在信息足够时锁定选项；信息仍不足时用简短口语追问。

规则（必须遵守）：
1. 只根据用户**已经明确表达或可合理、无歧义推断**的信息作答；不足则追问，不要猜测。
2. 只有能唯一对应某一题的**某一个选项**时，才在 commits 里提交；choice 必须与题目 options 中的某一项**逐字一致**（含标点、空格）。
3. **补充与改口**：用户可随时用语音追加细节、改口、否定先前说法。以**最新且明确**的内容为准；若新内容与已锁定答案冲突，必须用 commits **覆盖**该题（例如用户说「刚才说错了」「年龄那题改成…」「上一题不算」等）。
4. 一回合可提交 0 题或多题；已完成的题仍可在后续回合被用户纠正并再次 commit。
5. reply 用简短自然的中文，适合**语音朗读**；不要输出 JSON/字段名。

问卷与进度：
{_format_questionnaire(questionnaire, answers)}

未完成题 id 列表（便于你聚焦）：{json.dumps(pending, ensure_ascii=False)}
已锁定答案（可被取消或覆盖）：{json.dumps(done, ensure_ascii=False)}
"""


def _coerce_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _coerce_agent_turn_from_text(text: str) -> AgentTurnOutput:
    raw = (text or "").strip()
    if not raw:
        return AgentTurnOutput(reply="我没收到有效内容，请用一句话再说一下。", commits=[])
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return AgentTurnOutput(reply=raw[:2000], commits=[])
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return AgentTurnOutput(reply="我没能解析你的需求，请换个说法再说一次。", commits=[])
    try:
        return AgentTurnOutput.model_validate(data)
    except Exception:
        reply = data.get("reply") if isinstance(data, dict) else None
        if isinstance(reply, str):
            return AgentTurnOutput(reply=reply, commits=[])
        return AgentTurnOutput(reply="回复格式有点问题，请再发一条消息。", commits=[])


def _fallback_json_agent_turn(system: str, messages: list[BaseMessage]) -> AgentTurnOutput:
    suffix = (
        "\n\n【输出格式】只输出一个 JSON 对象（不要用 markdown 代码围栏），"
        '形如 {"reply":"对用户说的中文","commits":[{"question_id":"题id","choice":"选项原文"}]}。'
        "若本回合不提交任何题，commits 必须为 []。"
    )
    plain = get_chat_llm()
    resp = plain.invoke([SystemMessage(content=system + suffix), *messages])
    text = _coerce_message_text(getattr(resp, "content", None))
    return _coerce_agent_turn_from_text(text)


def _validate_commits(
    questionnaire: list[dict[str, Any]],
    parsed: AgentTurnOutput,
) -> dict[str, str]:
    by_id = {q["id"]: q for q in questionnaire}
    out: dict[str, str] = {}
    for c in parsed.commits:
        q = by_id.get(c.question_id)
        if not q:
            continue
        if c.choice in q["options"]:
            out[c.question_id] = c.choice
    return out


def agent_node(state: AgentState) -> dict[str, Any]:
    questionnaire = state["questionnaire"]
    answers = dict(state.get("answers") or {})
    messages: list[BaseMessage] = list(state["messages"])

    system = build_system_prompt(questionnaire, answers)
    lc_messages: list[BaseMessage] = [SystemMessage(content=system), *messages]
    parsed: AgentTurnOutput | None = None
    try:
        structured = get_structured_agent_llm()
        parsed = structured.invoke(lc_messages)
    except Exception:
        parsed = None
    if parsed is None:
        parsed = _fallback_json_agent_turn(system, messages)
    new_answers = _validate_commits(questionnaire, parsed)
    return {
        "messages": [AIMessage(content=parsed.reply)],
        "answers": new_answers,
    }


def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(checkpointer=MemorySaver())


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_agent_graph()
    return _compiled
