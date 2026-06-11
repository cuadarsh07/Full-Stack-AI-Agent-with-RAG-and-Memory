from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


AgentCallable = Callable[[str, Any, Any], Awaitable[Any]]
SynthesizerCallable = Callable[[str, Optional[str], Optional[str], List[str]], Awaitable[str]]


class AgentState(TypedDict, total=False):
    # State is the graph's shared memory: every node reads and writes this object.
    user_message: str
    chat_history: list
    route: Optional[str]
    resume_context: Optional[str]
    web_context: Optional[str]
    final_answer: Optional[str]
    needs_resume: bool
    needs_web: bool
    needs_more_info: bool
    tool_calls: List[str]
    retry_count: int
    error: Optional[str]

    # Runtime dependencies injected by main.py to avoid circular imports.
    session: Any
    telemetry: Any
    resume_agent: AgentCallable
    web_agent: AgentCallable
    synthesizer: SynthesizerCallable
    tool_payloads: List[Any]


RESUME_INTENT_RE = re.compile(
    r"\b("
    r"adarsh|resume|cv|profile|portfolio|background|career|candidate|"
    r"skill|skills|project|projects|education|college|degree|experience|"
    r"internship|work|built|ai project|machine learning|ml|developer"
    r")\b",
    re.IGNORECASE,
)

WEB_INTENT_RE = re.compile(
    r"\b("
    r"latest|current|recent|today|tonight|news|live|now|happening|"
    r"market|trend|trends|trending|updates|2026|this week|this month"
    r")\b",
    re.IGNORECASE,
)

SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|heya|yo|thanks|thank you|bye|goodbye|how are you|what'?s up|"
    r"what is your name|what'?s your name|who are you|what can you do|help|help me|how can you help)"
    r"[!.?\s]*$",
    re.IGNORECASE,
)


def _tool_calls(state: AgentState) -> List[str]:
    return list(state.get("tool_calls") or [])


def _tool_payloads(state: AgentState) -> List[Any]:
    return list(state.get("tool_payloads") or [])


async def supervisor_node(state: AgentState) -> AgentState:
    # Nodes are workers: this one chooses which specialist should run.
    message = state.get("user_message", "")
    needs_resume = bool(RESUME_INTENT_RE.search(message))
    needs_web = bool(WEB_INTENT_RE.search(message))
    lowered = message.lower()

    if needs_web and re.search(r"\bai trends?\b|\btrend matters most\b", lowered) and not re.search(
        r"\b(adarsh|resume|cv|profile|portfolio|candidate|his skills|adarsh's)\b",
        lowered,
    ):
        needs_resume = False

    if needs_resume and needs_web:
        route = "hybrid"
    elif needs_resume:
        route = "resume"
    elif needs_web:
        route = "web"
    elif SMALLTALK_RE.match(message):
        route = "smalltalk"
    else:
        route = "smalltalk"

    return {
        **state,
        "route": route,
        "needs_resume": needs_resume,
        "needs_web": needs_web,
        "needs_more_info": False,
        "tool_calls": _tool_calls(state),
        "tool_payloads": _tool_payloads(state),
        "retry_count": int(state.get("retry_count") or 0),
    }


async def resume_node(state: AgentState) -> AgentState:
    try:
        payload = await state["resume_agent"](
            state.get("user_message", ""),
            state.get("session"),
            state.get("telemetry"),
        )
        tool_calls = _tool_calls(state)
        tool_calls.append("resume_search")
        tool_payloads = _tool_payloads(state)
        tool_payloads.append(payload)

        if not getattr(payload, "success", False):
            error = getattr(getattr(payload, "error", None), "message", None) or "Resume search failed."
            return {
                **state,
                "resume_context": None,
                "tool_calls": tool_calls,
                "tool_payloads": tool_payloads,
                "error": error,
            }

        return {
            **state,
            "resume_context": (getattr(payload, "summary", None) or "").strip(),
            "tool_calls": tool_calls,
            "tool_payloads": tool_payloads,
        }
    except Exception as exc:
        return {**state, "error": f"Resume node failed: {exc}", "retry_count": int(state.get("retry_count") or 0) + 1}


async def web_node(state: AgentState) -> AgentState:
    try:
        payload = await state["web_agent"](
            state.get("user_message", ""),
            state.get("session"),
            state.get("telemetry"),
        )
        tool_calls = _tool_calls(state)
        tool_calls.append("live_web_search")
        tool_payloads = _tool_payloads(state)
        tool_payloads.append(payload)

        if not getattr(payload, "success", False):
            error = getattr(getattr(payload, "error", None), "message", None) or "Live web search failed."
            return {
                **state,
                "web_context": None,
                "tool_calls": tool_calls,
                "tool_payloads": tool_payloads,
                "error": error,
            }

        return {
            **state,
            "web_context": (getattr(payload, "summary", None) or "").strip(),
            "tool_calls": tool_calls,
            "tool_payloads": tool_payloads,
        }
    except Exception as exc:
        return {**state, "error": f"Web node failed: {exc}", "retry_count": int(state.get("retry_count") or 0) + 1}


async def hybrid_parallel_node(state: AgentState) -> AgentState:
    try:
        resume_payload, web_payload = await asyncio.gather(
            state["resume_agent"](
                state.get("user_message", ""),
                state.get("session"),
                state.get("telemetry"),
            ),
            state["web_agent"](
                state.get("user_message", ""),
                state.get("session"),
                state.get("telemetry"),
            ),
        )
    except Exception as exc:
        return {**state, "error": f"Hybrid node failed: {exc}", "retry_count": int(state.get("retry_count") or 0) + 1}

    tool_calls = _tool_calls(state)
    tool_calls.extend(["resume_search", "live_web_search"])
    tool_payloads = _tool_payloads(state)
    tool_payloads.extend([resume_payload, web_payload])

    updates: AgentState = {
        **state,
        "tool_calls": tool_calls,
        "tool_payloads": tool_payloads,
    }
    errors = []

    if getattr(resume_payload, "success", False):
        updates["resume_context"] = (getattr(resume_payload, "summary", None) or "").strip()
    else:
        errors.append(getattr(getattr(resume_payload, "error", None), "message", None) or "Resume search failed.")

    if getattr(web_payload, "success", False):
        updates["web_context"] = (getattr(web_payload, "summary", None) or "").strip()
    else:
        errors.append(getattr(getattr(web_payload, "error", None), "message", None) or "Live web search failed.")

    if errors:
        updates["error"] = " ".join(errors)

    return updates


async def smalltalk_node(state: AgentState) -> AgentState:
    message = state.get("user_message", "").strip().lower()
    if message.startswith(("thank", "thanks")):
        answer = "You're welcome. Happy to keep going with Adarsh's profile or a current tech topic."
    elif message.startswith(("bye", "goodbye")):
        answer = "Goodbye. Come back anytime if you want to continue."
    elif re.search(r"\b(name|who are you)\b", message):
        answer = "I'm Adarsh AI, your personal guide for Adarsh Kumar's resume, projects, skills, and current tech topics."
    elif "what can you do" in message or message.startswith("help") or "how can you help" in message:
        answer = "I can explain Adarsh Kumar's background, summarize his resume for recruiters, compare his skills with current roles, and check live web context."
    elif "how are you" in message or "what's up" in message:
        answer = "I'm doing well, bro. Ready to help with Adarsh's resume, projects, skills, or anything current you want checked."
    else:
        answer = "Hey, good to see you. I can help with Adarsh's resume, projects, skills, or current tech topics."

    return {**state, "final_answer": answer}


async def final_answer_node(state: AgentState) -> AgentState:
    resume_context = state.get("resume_context")
    web_context = state.get("web_context")
    tool_calls = _tool_calls(state)

    if resume_context and not web_context:
        answer = resume_context
    elif web_context and not resume_context:
        answer = web_context
    elif resume_context and web_context:
        answer = f"Resume side: {resume_context}\nCurrent AI/web side: {web_context}"
    else:
        error = state.get("error")
        answer = (
            "Escalation: I could not retrieve enough verified information to answer that safely."
            if error
            else "I do not have enough context to answer that yet."
        )

    return {**state, "final_answer": answer}


def route_after_supervisor(state: AgentState) -> str:
    # Conditional edges decide where the graph should go next.
    route = state.get("route")
    if route == "hybrid":
        return "hybrid"
    if route == "resume":
        return "resume"
    if route == "web":
        return "web"
    return "smalltalk"


def route_after_resume(state: AgentState) -> str:
    if state.get("needs_web"):
        return "web"
    return "final"


builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("resume_node", resume_node)
builder.add_node("web_node", web_node)
builder.add_node("hybrid_parallel_node", hybrid_parallel_node)
builder.add_node("smalltalk_node", smalltalk_node)
builder.add_node("final_answer_node", final_answer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "hybrid": "hybrid_parallel_node",
        "resume": "resume_node",
        "web": "web_node",
        "smalltalk": "smalltalk_node",
    },
)
builder.add_conditional_edges(
    "resume_node",
    route_after_resume,
    {
        "web": "web_node",
        "final": "final_answer_node",
    },
)
builder.add_edge("web_node", "final_answer_node")
builder.add_edge("hybrid_parallel_node", "final_answer_node")
builder.add_edge("smalltalk_node", END)
builder.add_edge("final_answer_node", END)

agent_graph = builder.compile()
