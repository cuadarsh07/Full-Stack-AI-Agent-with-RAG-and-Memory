import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Type

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
RAG_REWRITE_MODEL_NAME = "llama3-8b-8192"
MAX_AGENT_ITERATIONS = 5
ESCALATION_PREFIX = "Escalation:"
SMALL_TALK_MAX_LENGTH = 120
MAX_TOOL_RESULT_CHARS = 2500
AGENT_REQUEST_RETRIES = 2
AGENT_RETRY_DELAY_SECONDS = 2
AGENT_LOOP_DELAY_SECONDS = 1
MCP_TOOL_RETRIES = 1
MCP_TOOL_RETRY_DELAY_SECONDS = 0.3

# Required global MCP subprocess parameters.
server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
    env=os.environ.copy(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: connect with pure async context managers.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            app.state.mcp_session = session
            print("Successfully connected to MCP Server via clean async lifespan!")
            yield
    # Shutdown handled automatically by context manager exits.


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

SESSION_STORE: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SummaryRequest(StrictBaseModel):
    text: str = Field(min_length=1)
    style: str = Field(min_length=1)


class QuestionRequest(StrictBaseModel):
    question: str = Field(min_length=1)


class AgentRequest(StrictBaseModel):
    question: str = Field(min_length=1)


class MessageItem(StrictBaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ChatHistoryRequest(StrictBaseModel):
    messages: List[MessageItem] = Field(min_length=1)


class RunResumeAgentInput(StrictBaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="Question to delegate to the resume specialist agent.",
    )


class RunWebAgentInput(StrictBaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="Question to delegate to the web research specialist agent.",
    )


class ToolErrorPayload(StrictBaseModel):
    type: str
    message: str


class ToolExecutionPayload(StrictBaseModel):
    success: bool
    tool_name: str
    query: str
    summary: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[ToolErrorPayload] = None


class ToolUsageRecord(StrictBaseModel):
    name: str
    label: str
    query: str
    status: Literal["success", "error"]
    result_count: int = 0
    error: Optional[str] = None


class SupervisorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str = Field(
        min_length=1,
        description="The actual conversational response to the user, formatted in clean Markdown.",
    )
    suggested_follow_ups: List[str] = Field(
        min_length=2,
        max_length=3,
        description="A list of 2-3 natural follow-up questions the user might want to ask next.",
    )


class MasterChatRequest(StrictBaseModel):
    session_id: Optional[str] = None
    messages: List[MessageItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_last_message(self) -> "MasterChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("The last message in the conversation must be from the user.")
        return self


class MasterChatResponse(StrictBaseModel):
    session_id: str
    message_id: str
    answer: str
    structured_response: SupervisorResponse
    tools_used: List[ToolUsageRecord] = Field(default_factory=list)
    iterations: int
    escalated: bool = False
    warnings: List[str] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(StrictBaseModel):
    message_id: str = Field(min_length=1)
    thumbs_up: bool
    session_id: Optional[str] = None
    user_message: Optional[str] = None
    assistant_message: Optional[str] = None
    tools_used: List[ToolUsageRecord] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    label: str
    description: str
    input_model: Type[StrictBaseModel]
    executor: Callable[[StrictBaseModel, ClientSession], Awaitable[ToolExecutionPayload]]


@dataclass
class TelemetryState:
    started_at: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    agent_traces: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.agent_traces is None:
            self.agent_traces = []

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(prompt_tokens, 0)
        self.completion_tokens += max(completion_tokens, 0)

    def add_agent_trace(self, agent: str, latency_ms: float) -> None:
        self.agent_traces.append({"agent": agent, "latency_ms": round(latency_ms, 3)})

    def build_payload(self, total_latency_ms: float) -> Dict[str, Any]:
        total_tokens = self.prompt_tokens + self.completion_tokens
        return {
            "total_latency_ms": round(total_latency_ms, 3),
            "total_tokens": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": total_tokens,
            },
            "agent_traces": self.agent_traces,
        }


SMALL_TALK_PATTERNS = {
    "greeting": re.compile(
        r"^(?:hi|hello|hey|heya|yo|good\s+(?:morning|afternoon|evening)|hiya)[!.\s]*$",
        re.IGNORECASE,
    ),
    "farewell": re.compile(
        r"^(?:bye|goodbye|see\s+ya|see\s+you|talk\s+later|catch\s+you\s+later|farewell)[!.\s]*$",
        re.IGNORECASE,
    ),
    "thanks": re.compile(
        r"^(?:thanks|thank\s+you|tysm|thx|appreciate\s+it)[!.\s]*$",
        re.IGNORECASE,
    ),
    "small_talk": re.compile(
        r"^(?:how\s+are\s+you|how\'s\s+it\s+going|what\'s\s+up)[?.!\s]*$",
        re.IGNORECASE,
    ),
}


def get_groq_client() -> Groq:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return client


def extract_groq_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt_tokens, completion_tokens


def default_follow_ups() -> List[str]:
    return [
        "What parts of Adarsh's resume are most relevant to this?",
        "Can you compare this with the latest market context?",
    ]


def make_supervisor_response(final_answer: str, suggested_follow_ups: Optional[List[str]] = None) -> SupervisorResponse:
    clean_answer = final_answer.strip() or (
        f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
    )
    clean_follow_ups = [
        follow_up.strip()
        for follow_up in (suggested_follow_ups or default_follow_ups())
        if follow_up and follow_up.strip()
    ]

    while len(clean_follow_ups) < 2:
        clean_follow_ups.append(default_follow_ups()[len(clean_follow_ups)])

    return SupervisorResponse(
        final_answer=clean_answer,
        suggested_follow_ups=clean_follow_ups[:3],
    )


def parse_supervisor_response(raw_content: str) -> SupervisorResponse:
    try:
        parsed = json.loads(raw_content)
        return SupervisorResponse.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(f"Supervisor response did not match the required JSON schema: {exc}") from exc


def normalize_small_talk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def classify_small_talk(text: str) -> Optional[str]:
    normalized_text = normalize_small_talk_text(text)
    if not normalized_text or len(normalized_text) > SMALL_TALK_MAX_LENGTH:
        return None

    if normalized_text.lower() in {"sup", "good to see you"}:
        return "small_talk"

    for category, pattern in SMALL_TALK_PATTERNS.items():
        if pattern.fullmatch(normalized_text):
            return category

    return None


def build_small_talk_reply(text: str) -> Optional[str]:
    category = classify_small_talk(text)
    if category == "greeting":
        return (
            "Hi. I can help with Adarsh Kumar's background, projects, and resume, or look up "
            "current information on the web."
        )

    if category == "farewell":
        return "Goodbye. Come back anytime if you want to continue the conversation."

    if category == "thanks":
        return "You're welcome. If you want, you can ask about Adarsh's experience or any current topic next."

    if category == "small_talk":
        return "I'm doing well and ready to help. Ask about Adarsh's background or any live topic you want me to check."

    return None


def build_tool_schema(tool_spec: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_spec.name,
            "description": tool_spec.description,
            "parameters": tool_spec.input_model.model_json_schema(),
        },
    }


def build_tool_error(tool_name: str, query: str, error_type: str, message: str) -> ToolExecutionPayload:
    return ToolExecutionPayload(
        success=False,
        tool_name=tool_name,
        query=query,
        error=ToolErrorPayload(type=error_type, message=message),
    )


def truncate_text(value: Any, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text

    return f"{text[: limit - 3].rstrip()}..."


def parse_mcp_payload(raw_payload: str, expected_tool_name: str, query: str) -> ToolExecutionPayload:
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        return build_tool_error(
            tool_name=expected_tool_name,
            query=query,
            error_type="invalid_payload",
            message=f"MCP tool returned non-JSON payload: {exc}. Raw: {truncate_text(raw_payload)}",
        )

    try:
        payload = ToolExecutionPayload.model_validate(parsed)
    except ValidationError as exc:
        return build_tool_error(
            tool_name=expected_tool_name,
            query=query,
            error_type="invalid_payload",
            message=f"MCP tool returned invalid schema: {exc.json()}",
        )

    return payload


async def call_mcp_tool_payload(tool_name: str, query: str, session: ClientSession) -> ToolExecutionPayload:
    last_error: Optional[Exception] = None

    for attempt in range(MCP_TOOL_RETRIES + 1):
        try:
            result = await session.call_tool(tool_name, arguments={"query": query})
            text_chunks: List[str] = []
            for content_item in getattr(result, "content", []) or []:
                text_value = getattr(content_item, "text", None)
                if isinstance(text_value, str) and text_value.strip():
                    text_chunks.append(text_value)
                elif hasattr(content_item, "model_dump_json"):
                    text_chunks.append(content_item.model_dump_json())
                else:
                    text_chunks.append(str(content_item))

            raw_response = "\n".join(text_chunks).strip() or str(result)
            return parse_mcp_payload(raw_response, tool_name, query)
        except Exception as exc:
            last_error = exc
            if attempt >= MCP_TOOL_RETRIES:
                break
            await asyncio.sleep(MCP_TOOL_RETRY_DELAY_SECONDS)

    return build_tool_error(
        tool_name=tool_name,
        query=query,
        error_type="mcp_client_error",
        message=f"MCP call failed: {last_error}",
    )


def sanitize_rewritten_query(raw_query: str, fallback_query: str) -> str:
    rewritten_query = re.sub(r"\s+", " ", raw_query.strip().strip("\"'`")).strip()
    if not rewritten_query:
        return fallback_query

    return rewritten_query[:120]


def rewrite_resume_search_query(query: str, telemetry: Optional[TelemetryState] = None) -> str:
    try:
        rewrite_response = get_groq_client().chat.completions.create(
            model=RAG_REWRITE_MODEL_NAME,
            temperature=0,
            max_tokens=32,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a RAG Search Optimizer. Take the user's input query and rewrite it "
                        "into a short string of 3-5 high-relevance technical keywords optimized for "
                        "vector database semantic search. Return ONLY the search terms, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        if telemetry is not None:
            prompt_tokens, completion_tokens = extract_groq_usage(rewrite_response)
            telemetry.add_usage(prompt_tokens, completion_tokens)

        raw_rewritten_query = rewrite_response.choices[0].message.content or ""
        return sanitize_rewritten_query(raw_rewritten_query, query)
    except Exception:
        return query


async def run_resume_agent(
    query: str,
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    started_at = time.perf_counter()
    rewritten_query = rewrite_resume_search_query(query, telemetry)
    tool_payload = await call_mcp_tool_payload("search_resume_database", rewritten_query, session)
    if not tool_payload.success:
        if telemetry is not None:
            telemetry.add_agent_trace("resume_agent", (time.perf_counter() - started_at) * 1000)
        if tool_payload.error is None:
            return build_tool_error(
                tool_name="run_resume_agent",
                query=query,
                error_type="tool_execution_error",
                message="Resume specialist failed without an explicit error.",
            )

        return build_tool_error(
            tool_name="run_resume_agent",
            query=query,
            error_type=tool_payload.error.type,
            message=tool_payload.error.message,
        )

    context_lines = []
    for item in tool_payload.results:
        source = item.get("source") or "unknown_source"
        page = item.get("page")
        location = f"{source} page {page}" if page is not None else str(source)
        context_lines.append(f"- [{location}] {item.get('content', '')}")

    context_text = "\n".join(context_lines) if context_lines else "No resume matches were found."

    try:
        specialist_response = get_groq_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Resume Specialist. Your only job is to analyze Adarsh's resume "
                        "data to answer specific questions about his background, skills, and projects."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Resume evidence:\n{context_text}\n\n"
                        "Answer using only the evidence above. If evidence is insufficient, say so clearly."
                    ),
                },
            ],
        )
        if telemetry is not None:
            prompt_tokens, completion_tokens = extract_groq_usage(specialist_response)
            telemetry.add_usage(prompt_tokens, completion_tokens)
        summary = (specialist_response.choices[0].message.content or "").strip()
    except Exception as exc:
        if telemetry is not None:
            telemetry.add_agent_trace("resume_agent", (time.perf_counter() - started_at) * 1000)
        return build_tool_error(
            tool_name="run_resume_agent",
            query=query,
            error_type="worker_model_error",
            message=f"Resume specialist model call failed: {exc}",
        )

    if not summary:
        summary = "The resume specialist could not produce a grounded answer from the retrieved data."

    if telemetry is not None:
        telemetry.add_agent_trace("resume_agent", (time.perf_counter() - started_at) * 1000)

    return ToolExecutionPayload(
        success=True,
        tool_name="run_resume_agent",
        query=query,
        summary=summary,
        results=tool_payload.results,
    )


async def run_web_agent(
    query: str,
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    started_at = time.perf_counter()
    tool_payload = await call_mcp_tool_payload("search_live_web", query, session)
    if not tool_payload.success:
        if telemetry is not None:
            telemetry.add_agent_trace("web_agent", (time.perf_counter() - started_at) * 1000)
        if tool_payload.error is None:
            return build_tool_error(
                tool_name="run_web_agent",
                query=query,
                error_type="tool_execution_error",
                message="Web specialist failed without an explicit error.",
            )

        return build_tool_error(
            tool_name="run_web_agent",
            query=query,
            error_type=tool_payload.error.type,
            message=tool_payload.error.message,
        )

    context_lines = []
    for item in tool_payload.results:
        url = item.get("url") or "unknown_url"
        context_lines.append(f"- [{url}] {item.get('content', '')}")

    context_text = "\n".join(context_lines) if context_lines else "No live web matches were found."

    try:
        specialist_response = get_groq_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Web Research Specialist. Your only job is to find the latest "
                        "real-time information from the live internet."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Live web evidence:\n{context_text}\n\n"
                        "Answer using only the evidence above. If evidence is insufficient, say so clearly."
                    ),
                },
            ],
        )
        if telemetry is not None:
            prompt_tokens, completion_tokens = extract_groq_usage(specialist_response)
            telemetry.add_usage(prompt_tokens, completion_tokens)
        summary = (specialist_response.choices[0].message.content or "").strip()
    except Exception as exc:
        if telemetry is not None:
            telemetry.add_agent_trace("web_agent", (time.perf_counter() - started_at) * 1000)
        return build_tool_error(
            tool_name="run_web_agent",
            query=query,
            error_type="worker_model_error",
            message=f"Web specialist model call failed: {exc}",
        )

    if not summary:
        summary = "The web specialist could not produce a grounded answer from the retrieved sources."

    if telemetry is not None:
        telemetry.add_agent_trace("web_agent", (time.perf_counter() - started_at) * 1000)

    return ToolExecutionPayload(
        success=True,
        tool_name="run_web_agent",
        query=query,
        summary=summary,
        results=tool_payload.results,
    )


async def run_resume_agent_executor(
    tool_input: RunResumeAgentInput,
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    return await run_resume_agent(tool_input.query, session, telemetry)


async def run_web_agent_executor(
    tool_input: RunWebAgentInput,
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    return await run_web_agent(tool_input.query, session, telemetry)


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "run_resume_agent": ToolSpec(
        name="run_resume_agent",
        label="Consulted resume specialist",
        description="Delegate to the resume specialist worker for grounded answers from Adarsh's resume database.",
        input_model=RunResumeAgentInput,
        executor=run_resume_agent_executor,
    ),
    "run_web_agent": ToolSpec(
        name="run_web_agent",
        label="Consulted web specialist",
        description="Delegate to the web research specialist worker for current live internet information.",
        input_model=RunWebAgentInput,
        executor=run_web_agent_executor,
    ),
}

TOOLS_MENU = [build_tool_schema(tool_spec) for tool_spec in TOOL_REGISTRY.values()]


def as_groq_messages(messages: List[MessageItem]) -> List[Dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def build_master_system_prompt() -> str:
    return f"""You are a Supervisor / Project Manager assistant coordinating specialist workers.
Rules:
1. Use run_resume_agent for questions about the resume, candidate profile, projects, skills, education, or past experience.
2. Use run_web_agent for current events, recent facts, or information that may have changed after training.
3. If the user is just saying hello, goodbye, thanking you, or making small talk, DO NOT use any worker agents. Reply directly and conversationally from your own pre-trained knowledge.
4. NEVER output raw XML, tool markup, or <function> tags under any circumstances.
5. You may call worker agents multiple times when needed, but keep each call focused.
6. Never invent worker outputs. If a worker fails, either try the other worker or explain the limitation.
7. If you still do not have enough verified information after using workers, start your answer with '{ESCALATION_PREFIX}' and clearly state what is missing.
8. Give the final answer directly and concisely. Do not narrate your internal chain of thought.
9. Prefer grounded answers over broad speculation.
10. When producing a final response for the user, return ONLY a valid JSON object matching this exact shape:
{{
  "final_answer": "Clean Markdown answer for the user",
  "suggested_follow_ups": ["Natural follow-up question 1", "Natural follow-up question 2"]
}}
11. suggested_follow_ups must contain 2-3 concise, natural questions. Do not include any keys outside this JSON structure.
"""


def get_or_create_session_id(session_id: Optional[str]) -> str:
    return session_id or str(uuid.uuid4())


def hydrate_visible_messages(session_id: str, incoming_messages: List[MessageItem]) -> List[MessageItem]:
    with SESSION_LOCK:
        existing_session = SESSION_STORE.get(session_id)

    if not existing_session:
        return incoming_messages

    stored_messages = existing_session.get("messages", [])
    if len(incoming_messages) == 1 and incoming_messages[0].role == "user":
        return [*stored_messages, incoming_messages[0]]

    return incoming_messages


def store_session_messages(session_id: str, messages: List[MessageItem]) -> None:
    with SESSION_LOCK:
        SESSION_STORE[session_id] = {
            "messages": messages,
            "updated_at": time.time(),
        }


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    log_line = {"event": event_type, **payload}
    print(json.dumps(log_line, default=str))


def make_tool_usage_record(tool_name: str, payload: ToolExecutionPayload) -> ToolUsageRecord:
    tool_spec = TOOL_REGISTRY[tool_name]
    return ToolUsageRecord(
        name=tool_name,
        label=tool_spec.label,
        query=payload.query,
        status="success" if payload.success else "error",
        result_count=len(payload.results),
        error=payload.error.message if payload.error else None,
    )


async def execute_tool_call(
    tool_call: Any,
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> tuple[ToolExecutionPayload, ToolUsageRecord]:
    tool_name = tool_call.function.name
    raw_arguments = tool_call.function.arguments or "{}"
    tool_spec = TOOL_REGISTRY.get(tool_name)

    if tool_spec is None:
        payload = build_tool_error(
            tool_name=tool_name,
            query="",
            error_type="unknown_tool",
            message=f"Unknown tool requested: {tool_name}",
        )
        return payload, ToolUsageRecord(
            name=tool_name,
            label="Unknown tool",
            query="",
            status="error",
            error=payload.error.message if payload.error else None,
        )

    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        payload = build_tool_error(
            tool_name=tool_name,
            query="",
            error_type="validation_error",
            message=f"Tool arguments were not valid JSON: {exc}",
        )
        return payload, make_tool_usage_record(tool_name, payload)

    try:
        validated_input = tool_spec.input_model.model_validate(parsed_arguments)
    except ValidationError as exc:
        query = parsed_arguments.get("query", "") if isinstance(parsed_arguments, dict) else ""
        payload = build_tool_error(
            tool_name=tool_name,
            query=query,
            error_type="validation_error",
            message=exc.json(),
        )
        return payload, make_tool_usage_record(tool_name, payload)

    payload = await tool_spec.executor(validated_input, session, telemetry)
    return payload, make_tool_usage_record(tool_name, payload)


def format_assistant_tool_message(response_message: Any) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": response_message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_message.tool_calls or []
        ],
    }


def finalize_master_chat_response(
    session_id: str,
    final_message_id: str,
    visible_messages: List[MessageItem],
    answer: str,
    tools_used: List[ToolUsageRecord],
    iterations: int,
    warnings: List[str],
    escalated: bool,
    structured_response: Optional[SupervisorResponse] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    event_name: str = "master_chat_completed",
    extra_log_fields: Optional[Dict[str, Any]] = None,
) -> MasterChatResponse:
    response_payload = structured_response or make_supervisor_response(answer)
    answer = response_payload.final_answer
    assistant_message = MessageItem(role="assistant", content=answer)
    store_session_messages(session_id, [*visible_messages, assistant_message])
    log_event(
        event_name,
        {
            "session_id": session_id,
            "message_id": final_message_id,
            "structured_response": response_payload.model_dump(),
            "iterations": iterations,
            "tools_used": [tool.model_dump() for tool in tools_used],
            "escalated": escalated,
            "telemetry": telemetry or {},
            **(extra_log_fields or {}),
        },
    )

    print(
        json.dumps(
            {
                "event": "chat_transaction_telemetry",
                "session_id": session_id,
                "message_id": final_message_id,
                "telemetry": telemetry or {},
            },
            indent=2,
        )
    )

    return MasterChatResponse(
        session_id=session_id,
        message_id=final_message_id,
        answer=answer,
        structured_response=response_payload,
        tools_used=tools_used,
        iterations=iterations,
        escalated=escalated,
        warnings=warnings,
        telemetry=telemetry or {},
    )


def request_agent_step(groq_client: Groq, llm_messages: List[Dict[str, Any]]) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(AGENT_REQUEST_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": build_master_system_prompt()}, *llm_messages],
                tools=TOOLS_MENU,
                tool_choice="auto",
            )
        except Exception as exc:
            last_error = exc
            if attempt >= AGENT_REQUEST_RETRIES:
                break
            time.sleep(AGENT_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"Groq request failed after retries: {last_error}") from last_error


def request_structured_supervisor_response(
    groq_client: Groq,
    llm_messages: List[Dict[str, Any]],
    draft_answer: Optional[str] = None,
) -> Any:
    final_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{build_master_system_prompt()}\n"
                "You are now producing the final response. Do not mention tools unless the result is relevant. "
                "Return ONLY the required JSON object."
            ),
        },
        *llm_messages,
    ]

    if draft_answer:
        final_messages.append(
            {
                "role": "assistant",
                "content": draft_answer,
            }
        )
        final_messages.append(
            {
                "role": "user",
                "content": "Convert the draft answer into the required JSON object without adding unsupported claims.",
            }
        )

    last_error: Optional[Exception] = None
    for attempt in range(AGENT_REQUEST_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=final_messages,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            last_error = exc
            if attempt >= AGENT_REQUEST_RETRIES:
                break
            time.sleep(AGENT_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"Groq structured response failed after retries: {last_error}") from last_error


async def append_tool_results(
    session_id: str,
    iteration: int,
    llm_messages: List[Dict[str, Any]],
    tool_calls: List[Any],
    tools_used: List[ToolUsageRecord],
    warnings: List[str],
    session: ClientSession,
    telemetry: Optional[TelemetryState] = None,
) -> None:
    for tool_call in tool_calls:
        payload, usage_record = await execute_tool_call(tool_call, session, telemetry)
        tools_used.append(usage_record)

        if not payload.success and payload.error:
            warnings.append(f"{payload.tool_name}: {payload.error.message}")

        log_event(
            "tool_executed",
            {
                "session_id": session_id,
                "iteration": iteration,
                "tool": usage_record.model_dump(),
            },
        )

        llm_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": truncate_text(payload.model_dump_json()),
            }
        )


async def run_master_agent(request: MasterChatRequest, session: ClientSession) -> MasterChatResponse:
    telemetry = TelemetryState(started_at=time.perf_counter())
    session_id = get_or_create_session_id(request.session_id)
    visible_messages = hydrate_visible_messages(session_id, request.messages)
    llm_messages: List[Dict[str, Any]] = as_groq_messages(visible_messages)
    tools_used: List[ToolUsageRecord] = []
    warnings: List[str] = []
    final_message_id = str(uuid.uuid4())
    small_talk_reply = build_small_talk_reply(visible_messages[-1].content)

    if small_talk_reply is not None:
        total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
        return finalize_master_chat_response(
            session_id=session_id,
            final_message_id=final_message_id,
            visible_messages=visible_messages,
            answer=small_talk_reply,
            tools_used=tools_used,
            iterations=0,
            warnings=warnings,
            escalated=False,
            telemetry=telemetry.build_payload(total_latency_ms),
            event_name="master_chat_small_talk",
        )

    try:
        groq_client = get_groq_client()
    except RuntimeError as exc:
        answer = f"{ESCALATION_PREFIX} {exc}"
        total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
        return finalize_master_chat_response(
            session_id=session_id,
            final_message_id=final_message_id,
            visible_messages=visible_messages,
            answer=answer,
            tools_used=tools_used,
            iterations=0,
            warnings=[str(exc)],
            escalated=True,
            telemetry=telemetry.build_payload(total_latency_ms),
        )

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        try:
            response = request_agent_step(groq_client, llm_messages)
            prompt_tokens, completion_tokens = extract_groq_usage(response)
            telemetry.add_usage(prompt_tokens, completion_tokens)
        except Exception as exc:
            answer = f"{ESCALATION_PREFIX} I could not complete the request because the model call failed."
            warnings.append(f"Groq request failed: {exc}")
            total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
            return finalize_master_chat_response(
                session_id=session_id,
                final_message_id=final_message_id,
                visible_messages=visible_messages,
                answer=answer,
                tools_used=tools_used,
                iterations=iteration,
                warnings=warnings,
                escalated=True,
                telemetry=telemetry.build_payload(total_latency_ms),
                event_name="master_chat_failed",
                extra_log_fields={"error": str(exc)},
            )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []

        if not tool_calls:
            draft_answer = (response_message.content or "").strip()
            try:
                structured_response_call = request_structured_supervisor_response(
                    groq_client,
                    llm_messages,
                    draft_answer=draft_answer or None,
                )
                prompt_tokens, completion_tokens = extract_groq_usage(structured_response_call)
                telemetry.add_usage(prompt_tokens, completion_tokens)
                raw_answer = (structured_response_call.choices[0].message.content or "").strip()
            except Exception as exc:
                raw_answer = draft_answer
                warnings.append(f"Structured supervisor response failed: {exc}")

            if not raw_answer:
                answer = f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
                warnings.append("The model returned an empty final answer.")
                supervisor_response = make_supervisor_response(answer)
            else:
                try:
                    supervisor_response = parse_supervisor_response(raw_answer)
                    answer = supervisor_response.final_answer
                except ValueError as exc:
                    answer = raw_answer
                    supervisor_response = make_supervisor_response(answer)
                    warnings.append(str(exc))

            escalated = answer.startswith(ESCALATION_PREFIX)
            total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
            return finalize_master_chat_response(
                session_id=session_id,
                final_message_id=final_message_id,
                visible_messages=visible_messages,
                answer=answer,
                tools_used=tools_used,
                iterations=iteration,
                warnings=warnings,
                escalated=escalated,
                structured_response=supervisor_response,
                telemetry=telemetry.build_payload(total_latency_ms),
            )

        llm_messages.append(format_assistant_tool_message(response_message))
        await append_tool_results(
            session_id,
            iteration,
            llm_messages,
            tool_calls,
            tools_used,
            warnings,
            session,
            telemetry,
        )
        await asyncio.sleep(AGENT_LOOP_DELAY_SECONDS)

    answer = (
        f"{ESCALATION_PREFIX} I reached the maximum number of tool steps for this request and "
        "do not yet have enough verified information to answer confidently."
    )
    warnings.append("The agent loop hit the iteration limit.")
    total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
    return finalize_master_chat_response(
        session_id=session_id,
        final_message_id=final_message_id,
        visible_messages=visible_messages,
        answer=answer,
        tools_used=tools_used,
        iterations=MAX_AGENT_ITERATIONS,
        warnings=warnings,
        escalated=True,
        telemetry=telemetry.build_payload(total_latency_ms),
        extra_log_fields={"warning": "iteration_limit"},
    )


@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    system_prompt = f"""You are an expert editor. Summarize the text provided by the user.
    Use this exact style: {request.style}.
    You MUST respond in valid JSON format.
    Your JSON must contain exactly two keys:
    'title' (a short, catchy title for the text) and
    'content' (the actual summary)."""

    chat_completion = get_groq_client().chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.text},
        ],
        model=MODEL_NAME,
        response_format={"type": "json_object"},
    )

    raw_response = chat_completion.choices[0].message.content
    structured_summary = json.loads(raw_response)
    return structured_summary


@app.post("/ask")
async def ask_document(http_request: Request, request: QuestionRequest):
    session: ClientSession = http_request.app.state.mcp_session
    tool_result = await call_mcp_tool_payload("search_resume_database", request.question, session)
    docs = tool_result.results
    context_text = "\n\n".join([doc.get("content", "") for doc in docs])

    system_prompt = f"""You are a helpful company assistant.
    Answer the user's question using ONLY the following context.
    If the answer is not contained in the context, say exactly: 'I am sorry, but I do not have information about that in my documents.'

    Context:
    {context_text}
    """

    chat_completion = get_groq_client().chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question},
        ],
        model=MODEL_NAME,
    )

    return {
        "answer": chat_completion.choices[0].message.content,
        "sources_used": [doc.get("content", "") for doc in docs],
        "tool_success": tool_result.success,
        "tool_error": tool_result.error.model_dump() if tool_result.error else None,
    }


@app.post("/agent")
async def run_agent(http_request: Request, request: AgentRequest):
    session: ClientSession = http_request.app.state.mcp_session
    master_response = await run_master_agent(
        MasterChatRequest(messages=[MessageItem(role="user", content=request.question)]),
        session,
    )

    web_tool = next((tool for tool in master_response.tools_used if tool.name == "run_web_agent"), None)

    return {
        "answer": master_response.answer,
        "used_tool": web_tool is not None,
        "search_query": web_tool.query if web_tool else None,
    }


@app.post("/chat")
def run_chat(request: ChatHistoryRequest):
    conversation_history = as_groq_messages(request.messages)

    response = get_groq_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a friendly, conversational AI. You have perfect memory of this conversation.",
            },
            *conversation_history,
        ],
    )

    return {"reply": response.choices[0].message.content}


@app.post("/master-chat", response_model=MasterChatResponse)
async def master_chat(http_request: Request, request: MasterChatRequest) -> MasterChatResponse:
    session: ClientSession = http_request.app.state.mcp_session
    return await run_master_agent(request, session)


@app.post("/feedback")
def log_feedback(request: FeedbackRequest):
    log_event(
        "feedback_received",
        {
            "message_id": request.message_id,
            "thumbs_up": request.thumbs_up,
            "session_id": request.session_id,
            "user_message": request.user_message,
            "assistant_message": request.assistant_message,
            "tools_used": [tool.model_dump() for tool in request.tools_used],
        },
    )
    return {"status": "ok"}
