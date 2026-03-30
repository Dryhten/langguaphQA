from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(_BACKEND_DIR / ".env")

from graph import get_graph
from models import ChatMessageBody, SessionCreate
from state import questionnaire_to_dicts

app = FastAPI(title="Voice Questionnaire Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_questionnaires: dict[str, list] = {}
_graph = get_graph()


@app.post("/sessions")
def create_session(body: SessionCreate):
    sid = str(uuid.uuid4())
    session_questionnaires[sid] = questionnaire_to_dicts(body.questions)
    return {"session_id": sid}


@app.post("/sessions/{session_id}/messages")
def post_message(session_id: str, body: ChatMessageBody):
    if session_id not in session_questionnaires:
        raise HTTPException(status_code=404, detail="session not found")
    config = {"configurable": {"thread_id": session_id}}
    state = _graph.get_state(config)
    values = dict(state.values) if state.values else {}
    payload: dict = {"messages": [HumanMessage(content=body.text)]}
    if not values.get("questionnaire"):
        payload["questionnaire"] = session_questionnaires[session_id]
        payload["answers"] = {}
    try:
        out = _graph.invoke(payload, config)
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"model config error: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"model invoke failed: {e!s}") from e
    messages = out.get("messages") or []
    if not messages:
        raise HTTPException(status_code=500, detail="empty model response")
    last = messages[-1]
    assistant_text = getattr(last, "content", None) or str(last)
    return {"assistant_text": assistant_text, "answers": out.get("answers") or {}}


_root = Path(__file__).resolve().parent.parent
_frontend_index = _root / "frontend" / "index.html"
_survey_default = _root / "frontend" / "survey-default.json"


@app.get("/survey-default.json")
def serve_survey_default():
    if _survey_default.is_file():
        return FileResponse(_survey_default, media_type="application/json")
    raise HTTPException(status_code=404, detail="frontend/survey-default.json not found")


@app.get("/")
def serve_ui():
    if _frontend_index.is_file():
        return FileResponse(_frontend_index)
    raise HTTPException(status_code=404, detail="frontend/index.html not found")
