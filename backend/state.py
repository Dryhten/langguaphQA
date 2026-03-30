from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from models import QuestionSpec


def merge_answers(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    base = dict(left or {})
    upd = dict(right or {})
    if not upd:
        return base
    return {**base, **upd}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    questionnaire: list[dict[str, Any]]
    answers: Annotated[dict[str, str], merge_answers]


def questionnaire_to_dicts(items: list[QuestionSpec]) -> list[dict[str, Any]]:
    return [q.model_dump() for q in items]
