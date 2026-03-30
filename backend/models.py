from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionSpec(BaseModel):
    id: str
    text: str
    options: list[str] = Field(min_length=1)


class SessionCreate(BaseModel):
    questions: list[QuestionSpec] = Field(min_length=1)


class AnswerCommit(BaseModel):
    question_id: str
    choice: str


class AgentTurnOutput(BaseModel):
    reply: str = Field(description="对用户说的自然语言回复，可含追问或总结。")
    commits: list[AnswerCommit] = Field(
        default_factory=list,
        description="仅当能唯一对应某题选项时提交；choice 必须与题目 options 中某项完全一致。",
    )


class ChatMessageBody(BaseModel):
    text: str = Field(min_length=1)
