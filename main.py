import asyncio
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Type

from dotenv import load_dotenv
from agent_graph import agent_graph
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
MCP_TOOL_TIMEOUT_SECONDS = float(os.environ.get("MCP_TOOL_TIMEOUT_SECONDS", "120"))
MCP_STARTUP_TIMEOUT_SECONDS = float(os.environ.get("MCP_STARTUP_TIMEOUT_SECONDS", "10"))
MCP_REQUIRED = os.environ.get("MCP_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
MCP_SERVER_PATH = Path(__file__).with_name("mcp_server.py").resolve()
REQUIRED_ENV_KEYS = [
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "TAVILY_API_KEY",
    "HF_TOKEN",
]

# Required global MCP subprocess parameters.
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(MCP_SERVER_PATH)],
    env=os.environ.copy(),
)


def log_mcp_startup(status: str, message: str, **extra: Any) -> None:
    print(json.dumps({"event": "mcp_startup", "status": status, "message": message, **extra}, default=str))


def get_env_status() -> Dict[str, bool]:
    return {key: bool(os.environ.get(key)) for key in REQUIRED_ENV_KEYS}


def collect_exception_details(exc: BaseException) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }

    if isinstance(exc, BaseExceptionGroup):
        details["sub_exceptions"] = [
            collect_exception_details(sub_exception)
            for sub_exception in exc.exceptions
        ]

    return details


async def initialize_mcp_session() -> tuple[AsyncExitStack, ClientSession]:
    stack = AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return stack, session
    except BaseException:
        await stack.aclose()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mcp_session = None
    app.state.mcp_status = "starting"
    app.state.mcp_error = None
    app.state.mcp_error_details = None
    app.state.mcp_server_command = [server_params.command, *server_params.args]
    app.state.mcp_env_status = get_env_status()
    mcp_stack: Optional[AsyncExitStack] = None

    try:
        # MCP startup is bounded so the HTTP app cannot hang indefinitely while the
        # subprocess loads dependencies or waits during its initialize handshake.
        mcp_stack, session = await asyncio.wait_for(
            initialize_mcp_session(),
            timeout=MCP_STARTUP_TIMEOUT_SECONDS,
        )
        app.state.mcp_session = session
        app.state.mcp_status = "ready"
        log_mcp_startup(
            "ready",
            "MCP server initialized successfully.",
            timeout_seconds=MCP_STARTUP_TIMEOUT_SECONDS,
            command=app.state.mcp_server_command,
            env=app.state.mcp_env_status,
        )
    except asyncio.TimeoutError as exc:
        app.state.mcp_status = "timeout"
        app.state.mcp_error = f"MCP startup timed out after {MCP_STARTUP_TIMEOUT_SECONDS:g} seconds."
        app.state.mcp_error_details = collect_exception_details(exc)
        log_mcp_startup(
            "timeout",
            app.state.mcp_error,
            required=MCP_REQUIRED,
            command=app.state.mcp_server_command,
            env=app.state.mcp_env_status,
            exception=app.state.mcp_error_details,
        )
        if MCP_REQUIRED:
            raise RuntimeError(app.state.mcp_error) from exc
    except Exception as exc:
        app.state.mcp_status = "failed"
        app.state.mcp_error = f"MCP startup failed: {exc}"
        app.state.mcp_error_details = collect_exception_details(exc)
        log_mcp_startup(
            "failed",
            app.state.mcp_error,
            required=MCP_REQUIRED,
            command=app.state.mcp_server_command,
            env=app.state.mcp_env_status,
            exception=app.state.mcp_error_details,
        )
        if MCP_REQUIRED:
            raise

    try:
        yield
    finally:
        if mcp_stack is not None:
            await mcp_stack.aclose()
            log_mcp_startup("stopped", "MCP server connection closed.")


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
    executor: Callable[[StrictBaseModel, Optional[ClientSession], Optional["TelemetryState"]], Awaitable[ToolExecutionPayload]]


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


@dataclass(frozen=True)
class FollowUpResolution:
    original_user_message: str
    resolved_user_message: str
    followup_intent: Optional[str]
    previous_topic: Optional[str]


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
        r"^(?:how\s+are\s+(?:you|u)(?:\s+(?:bro|buddy|man|dude))?|how\'s\s+it\s+going|what\'s\s+up)[?.!\s]*$",
        re.IGNORECASE,
    ),
    "identity": re.compile(
        r"^(?:what\s+is\s+your\s+name|what\'s\s+your\s+name|who\s+are\s+you|introduce\s+yourself)[?.!\s]*$",
        re.IGNORECASE,
    ),
    "capabilities": re.compile(
        r"^(?:what\s+can\s+you\s+do|help|help\s+me|how\s+can\s+you\s+help)[?.!\s]*$",
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
        "Show Adarsh's strongest resume evidence",
        "Compare Adarsh's skills with current backend and AI roles",
    ]


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


INTERNAL_EVIDENCE_PHRASES = [
    "Based on the provided resume evidence",
    "Based on the provided live web evidence",
    "Based on the provided outputs",
    "Based on the provided evidence",
    "Based on the provided context",
]


def clean_answer_text(text: str) -> str:
    cleaned = (
        text.replace("\r\n", "\n")
        .replace("’", "'")
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("–", "-")
        .replace("â", "-")
        .replace("â€“", "-")
        .replace("â™‚", "")
        .replace("â", "")
        .replace("â€™", "'")
        .replace("â€œ", "\"")
        .replace("â€", "\"")
    )
    cleaned = cleaned.replace("T ools", "Tools").replace("A WS", "AWS").replace("F eb", "Feb")
    for phrase in INTERNAL_EVIDENCE_PHRASES:
        cleaned = re.sub(re.escape(phrase) + r",?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\bthere is insufficient information to\b",
        "I don't have enough evidence to",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned.strip(" \n\t-")


def split_answer_points(text: str) -> List[str]:
    normalized = clean_answer_text(text)
    if not normalized:
        return []

    points: List[str] = []
    for line in normalized.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        line = clean_answer_text(line)
        if line:
            points.append(line)

    if len(points) <= 1:
        points = re.split(r"(?<=[.!?])\s+", compact_whitespace(normalized))

    return [
        compact_whitespace(clean_answer_text(point))
        for point in points
        if compact_whitespace(clean_answer_text(point))
    ]


def first_complete_sentence(text: str, limit: int = 220) -> str:
    point = compact_whitespace(clean_answer_text(text))
    if not point:
        return ""

    sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", point)
    sentence = sentence_match.group(1).strip() if sentence_match else point
    if len(sentence) <= limit:
        return sentence

    shortened = sentence[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    if not shortened.endswith((".", "!", "?")):
        shortened = f"{shortened}."
    return shortened


def trim_point(text: str, limit: int = 180) -> str:
    return first_complete_sentence(text, limit)


def detect_weak_resume_evidence(text: str) -> bool:
    lowered = text.lower()
    weak_phrases = [
        "insufficient",
        "not enough",
        "not explicitly",
        "do not see",
        "don't see",
        "does not mention",
        "doesn't mention",
        "no named",
        "no resume matches",
        "could not produce",
        "could not retrieve",
    ]
    return any(phrase in lowered for phrase in weak_phrases)


def format_answer_points(opening: str, points: List[str], max_points: int = 4) -> str:
    opening = first_complete_sentence(opening, limit=260)
    clean_points = []
    for point in points:
        cleaned = first_complete_sentence(point)
        if cleaned and cleaned.lower() != opening.lower() and cleaned not in clean_points:
            clean_points.append(cleaned)
        if len(clean_points) >= max_points:
            break

    if not clean_points:
        return opening

    bullets = "\n".join(f"- {point}" for point in clean_points)
    return f"{opening}\n{bullets}"


def labeled_bullet(label: str, text: str) -> str:
    point = first_complete_sentence(text)
    return f"{label}: {point}" if point else ""


def infer_opening_sentence(
    clean_answer: str,
    *,
    has_resume: bool,
    has_web: bool,
    query: str,
) -> str:
    if has_resume and has_web:
        if "skill" in query:
            return "Adarsh's strongest skills connect well with current AI and automation trends."
        return "Adarsh's profile has a practical connection to the current web context."

    if has_resume:
        if "skill" in query:
            return "Adarsh's strongest technical skills are the clearest fit in the resume evidence."
        if "project" in query:
            return "The resume evidence points to practical project and backend implementation work."
        return "Here is the clearest resume-backed answer."

    if has_web:
        return "Here is the clearest current web-backed answer."

    answer_sentence = first_complete_sentence(clean_answer)
    if answer_sentence and not answer_sentence.lower().startswith(
        ("resume side:", "current ai/web side:", "latest web trend:")
    ):
        return answer_sentence

    return answer_sentence or "Here is the short version."


def build_concise_answer(
    final_answer: str,
    *,
    user_query: Optional[str] = None,
    route: Optional[str] = None,
    resume_context: Optional[str] = None,
    web_context: Optional[str] = None,
) -> str:
    clean_answer = clean_answer_text(final_answer) or (
        f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
    )
    query = (user_query or "").lower()
    has_resume = bool(resume_context)
    has_web = bool(web_context)

    if clean_answer.startswith(ESCALATION_PREFIX) and not has_resume and not has_web:
        return clean_answer

    if has_resume and has_web:
        resume_points = split_answer_points(resume_context or clean_answer)
        web_points = split_answer_points(web_context or clean_answer)
        opening = infer_opening_sentence(clean_answer, has_resume=True, has_web=True, query=query)
        if "job requirement" in query or "job requirements" in query:
            connection_point = "Adarsh's backend, microservice, API, Docker, and cloud exposure are the strongest match areas."
        elif "skill" in query:
            connection_point = "The strongest overlap is backend engineering plus practical automation and tooling experience."
        else:
            fit_points = split_answer_points(clean_answer)
            connection_point = fit_points[0] if fit_points else "The resume evidence should be compared against the current role requirements."
        return format_answer_points(
            opening,
            [
                labeled_bullet("Resume side", resume_points[0]) if resume_points else "",
                labeled_bullet("Current AI/web side", web_points[0]) if web_points else "",
                labeled_bullet("How they connect", connection_point),
            ],
            max_points=3,
        )

    if has_resume or route == "resume":
        points = split_answer_points(resume_context or clean_answer)
        evidence_text = f"{clean_answer} {resume_context or ''}"
        if "ai" in query and ("project" in query or "projects" in query) and detect_weak_resume_evidence(evidence_text):
            return format_answer_points(
                "I don’t see named AI projects in the resume evidence yet.",
                [f"Closest related evidence: {point}" for point in points[:2]],
                max_points=2,
            )

        opening = infer_opening_sentence(clean_answer, has_resume=True, has_web=False, query=query)
        return format_answer_points(opening, points[:4], max_points=4)

    if has_web or route == "web":
        points = split_answer_points(web_context or clean_answer)
        opening = infer_opening_sentence(clean_answer, has_resume=False, has_web=True, query=query)
        return format_answer_points(opening, points[:4], max_points=4)

    if len(clean_answer) <= 220 and "\n" not in clean_answer:
        return clean_answer

    points = split_answer_points(clean_answer)
    opening = points[0] if points else clean_answer
    if len(points) <= 1:
        return first_complete_sentence(opening, limit=220)

    return format_answer_points(first_complete_sentence(opening, limit=140), points[1:4], max_points=3)


def build_dynamic_follow_ups(
    user_query: Optional[str] = None,
    *,
    route: Optional[str] = None,
    has_resume: bool = False,
    has_web: bool = False,
) -> List[str]:
    query = (user_query or "").lower()
    follow_ups: List[str] = []

    def add(item: str) -> None:
        if item not in follow_ups and len(follow_ups) < 3:
            follow_ups.append(item)

    if "ai" in query and ("project" in query or "projects" in query) and (has_resume or route == "resume"):
        add("Show Adarsh's strongest AI-related resume points")
        add("What AI project should Adarsh add to his portfolio?")
        add("Compare Adarsh's AI experience with current AI job requirements")
    elif "ai" in query and (has_web or route == "web"):
        add("Summarize the current AI trend context in 3 bullets")
        add("Which AI trend matters most for software developers right now?")
        add("Compare current AI trends with Adarsh's backend skills")
    elif has_resume and has_web:
        add("Which of Adarsh's resume points best match current market needs?")
        add("What should Adarsh improve next for backend and AI roles?")
        add("Turn this comparison into interview talking points for Adarsh")
    elif has_resume or route == "resume":
        if "skill" in query or "skills" in query:
            add("Show Adarsh's strongest technical skills")
            add("Which skills should Adarsh highlight first?")
            add("Compare Adarsh's Java and Spring Boot skills with current backend roles")
        elif "project" in query or "portfolio" in query:
            add("Show Adarsh's strongest project evidence")
            add("What project should Adarsh add next?")
            add("Turn Adarsh's project evidence into portfolio bullet points")
        else:
            add("Show the strongest resume evidence")
            add("Summarize this for a recruiter")
            add("Compare Adarsh's background with current backend and AI job requirements")
    elif has_web or route == "web":
        add("Summarize the current web findings in 3 bullets")
        add("Why does this current trend matter for software developers?")
        add("Compare this trend with Adarsh's backend and AI skills")
    else:
        add("Show Adarsh's resume highlights")
        add("Check latest backend and AI job-market context for Adarsh")

    while len(follow_ups) < 2:
        add(default_follow_ups()[len(follow_ups)])

    return follow_ups[:3]


def make_supervisor_response(
    final_answer: str,
    suggested_follow_ups: Optional[List[str]] = None,
    *,
    user_query: Optional[str] = None,
    route: Optional[str] = None,
    resume_context: Optional[str] = None,
    web_context: Optional[str] = None,
) -> SupervisorResponse:
    clean_answer = final_answer.strip() or (
        f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
    )
    clean_answer = build_concise_answer(
        clean_answer,
        user_query=user_query,
        route=route,
        resume_context=resume_context,
        web_context=web_context,
    )
    clean_follow_ups = [
        follow_up.strip()
        for follow_up in (
            suggested_follow_ups
            or build_dynamic_follow_ups(
                user_query,
                route=route,
                has_resume=bool(resume_context),
                has_web=bool(web_context),
            )
        )
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
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = re.sub(r"\bu\b", "you", normalized)
    return normalized


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
        return "Hey, good to see you. I can help with Adarsh's resume, projects, skills, or current tech topics."

    if category == "farewell":
        return "Goodbye. Come back anytime if you want to continue the conversation."

    if category == "thanks":
        return "You're welcome. Happy to keep going whenever you want to explore Adarsh's profile or a current topic."

    if category == "small_talk":
        return "I'm doing well, bro. Ready to help with Adarsh's resume, projects, skills, or anything current you want checked."

    if category == "identity":
        return "I'm Adarsh AI, your personal guide for Adarsh Kumar's resume, projects, skills, and current tech topics."

    if category == "capabilities":
        return "I can explain Adarsh Kumar's background, summarize his resume for recruiters, compare his skills with current roles, and check live web context."

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

    if isinstance(parsed, dict) and "success" not in parsed and parsed.get("ok") is False:
        error_value = parsed.get("error") or "MCP tool returned an error."
        error_message = error_value if isinstance(error_value, str) else json.dumps(error_value, default=str)
        return build_tool_error(
            tool_name=str(parsed.get("tool_name") or expected_tool_name),
            query=str(parsed.get("query") or query),
            error_type=str(parsed.get("error_type") or "mcp_tool_error"),
            message=error_message,
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


async def call_mcp_tool_payload(
    tool_name: str,
    query: str,
    session: Optional[ClientSession],
) -> ToolExecutionPayload:
    if session is None:
        return build_tool_error(
            tool_name=tool_name,
            query=query,
            error_type="mcp_unavailable",
            message=(
                "MCP tools are unavailable because the MCP server did not initialize during startup. "
                "Check server logs for the mcp_startup event."
            ),
        )

    last_error: Optional[Exception] = None

    for attempt in range(MCP_TOOL_RETRIES + 1):
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments={"query": query}),
                timeout=MCP_TOOL_TIMEOUT_SECONDS,
            )
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
        except asyncio.TimeoutError:
            return build_tool_error(
                tool_name=tool_name,
                query=query,
                error_type="mcp_tool_timeout",
                message=f"MCP tool '{tool_name}' timed out after {MCP_TOOL_TIMEOUT_SECONDS:g} seconds.",
            )
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


def keyword_search_query(query: str, limit: int = 8) -> str:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "his",
        "in",
        "is",
        "me",
        "of",
        "on",
        "or",
        "tell",
        "the",
        "to",
        "what",
        "with",
    }
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9+#./-]+", query.lower())
        if len(token) > 1 and token not in stopwords
    ]
    return " ".join(dict.fromkeys(tokens[:limit])) or query


def rewrite_resume_search_query(query: str, telemetry: Optional[TelemetryState] = None) -> str:
    lowered = query.lower()
    if any(term in lowered for term in ("background", "overview", "recruiter", "resume evidence", "strongest")):
        return "experience skills projects education backend java spring boot microservices"
    if "job requirement" in lowered or "job requirements" in lowered:
        return "skills experience projects backend java spring boot microservices docker cloud"
    return sanitize_rewritten_query(keyword_search_query(query), query)


CONTACT_METADATA_RE = re.compile(
    r"(@|github\.com|linkedin\.com|phone|\+\d|\b\d{10}\b)",
    re.IGNORECASE,
)

RESUME_SECTION_PRIORITY = {
    "experience": 0,
    "skills": 1,
    "projects": 2,
    "education": 3,
    "profile": 9,
}


def resume_result_content(item: Dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "").strip()


def resume_result_section(item: Dict[str, Any]) -> str:
    return str(item.get("section_name") or item.get("metadata", {}).get("section_name") or "").strip()


def is_contact_metadata_result(item: Dict[str, Any]) -> bool:
    section_name = resume_result_section(item).lower()
    content = resume_result_content(item)
    if section_name != "profile":
        return False
    return bool(CONTACT_METADATA_RE.search(content))


def is_useful_resume_result(item: Dict[str, Any], query: str) -> bool:
    if not resume_result_content(item):
        return False
    if "contact" not in query.lower() and is_contact_metadata_result(item):
        return False
    return True


def ordered_resume_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    useful_results = [item for item in results if is_useful_resume_result(item, query)]
    return sorted(
        useful_results,
        key=lambda item: (
            RESUME_SECTION_PRIORITY.get(resume_result_section(item).lower(), 5),
            int(item.get("metadata", {}).get("chunk_index", 9999) or 9999),
        ),
    )


def summarize_resume_results(query: str, results: List[Dict[str, Any]]) -> str:
    ordered_results = ordered_resume_results(query, results)
    if not ordered_results:
        return "No resume matches were found."

    points = []
    for item in ordered_results[:4]:
        section_name = resume_result_section(item)
        content = first_complete_sentence(resume_result_content(item), limit=220)
        if not content:
            continue
        section_key = section_name.lower()
        if section_key == "experience":
            point = f"His experience includes {content}"
        elif section_key == "skills":
            point = f"His strongest technical areas include {content}"
        elif section_key == "projects":
            point = f"His project work includes {content}"
        elif section_key == "education":
            point = f"His education includes {content}"
        elif section_key == "profile":
            point = f"His profile highlights {content}"
        else:
            point = content
        points.append(point)

    if not points:
        return "No resume matches were found."

    return "\n".join(f"- {point}" for point in points)


NOISY_WEB_RESULT_RE = re.compile(
    r"\b(facebook|fan page|meme|reddit|tiktok|instagram|pinterest|bro,|don't like that)\b",
    re.IGNORECASE,
)


def rewrite_web_search_query(query: str) -> str:
    lowered = query.lower()
    vague_web = any(term in lowered for term in ("this", "that", "these", "those", "web context", "latest context"))
    if "trend matters most" in lowered or ("ai trend" in lowered and "developer" in lowered):
        return "current AI agent coding automation trends most important for software developers 2026"
    if "why do current ai trends matter" in lowered or ("ai" in lowered and "matter" in lowered and "developer" in lowered):
        return "why current AI agent and software automation trends matter for software developers 2026"
    if "job requirement" in lowered or "job requirements" in lowered:
        return "current backend developer AI developer job requirements Java Spring Boot microservices Docker cloud 2026"
    if "adarsh" in lowered and ("skill" in lowered or "skills" in lowered):
        return "current backend developer AI engineer skills Java Spring Boot microservices Docker cloud 2026"
    if "june 11" in lowered or "11 june" in lowered:
        return "June 11 famous historical events observances today"
    if vague_web:
        return sanitize_rewritten_query(keyword_search_query(query, limit=12), "current software developer technology trends 2026")
    return sanitize_rewritten_query(query, query)


def summarize_web_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No live web matches were found."

    points = []
    for item in results[:8]:
        raw_content = str(item.get("content") or item.get("snippet") or item.get("title") or "")
        if NOISY_WEB_RESULT_RE.search(raw_content):
            continue
        content = first_complete_sentence(raw_content, limit=240)
        if content:
            points.append(content)
        if len(points) >= 3:
            break

    if not points:
        return "No live web matches were found."

    return "\n".join(f"- {point}" for point in points)


async def run_resume_agent(
    query: str,
    session: Optional[ClientSession],
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    started_at = time.perf_counter()
    if session is None:
        tool_payload = await call_mcp_tool_payload("search_resume_database", query, session)
        if telemetry is not None:
            telemetry.add_agent_trace("resume_agent", (time.perf_counter() - started_at) * 1000)
        return build_tool_error(
            tool_name="run_resume_agent",
            query=query,
            error_type=tool_payload.error.type if tool_payload.error else "mcp_unavailable",
            message=tool_payload.error.message if tool_payload.error else "MCP tools are unavailable.",
        )

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

    summary = summarize_resume_results(query, tool_payload.results)

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
    session: Optional[ClientSession],
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    started_at = time.perf_counter()
    rewritten_query = rewrite_web_search_query(query)
    tool_payload = await call_mcp_tool_payload("search_live_web", rewritten_query, session)
    if not tool_payload.success:
        if telemetry is not None:
            telemetry.add_agent_trace("web_agent", (time.perf_counter() - started_at) * 1000)
        if tool_payload.error is None:
            return build_tool_error(
                tool_name="run_web_agent",
                query=rewritten_query,
                error_type="tool_execution_error",
                message="Web specialist failed without an explicit error.",
            )

        return build_tool_error(
            tool_name="run_web_agent",
            query=rewritten_query,
            error_type=tool_payload.error.type,
            message=tool_payload.error.message,
        )

    summary = summarize_web_results(tool_payload.results)

    if telemetry is not None:
        telemetry.add_agent_trace("web_agent", (time.perf_counter() - started_at) * 1000)

    return ToolExecutionPayload(
        success=True,
        tool_name="run_web_agent",
        query=rewritten_query,
        summary=summary,
        results=tool_payload.results,
    )


async def run_resume_agent_executor(
    tool_input: RunResumeAgentInput,
    session: Optional[ClientSession],
    telemetry: Optional[TelemetryState] = None,
) -> ToolExecutionPayload:
    return await run_resume_agent(tool_input.query, session, telemetry)


async def run_web_agent_executor(
    tool_input: RunWebAgentInput,
    session: Optional[ClientSession],
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


FOLLOW_UP_CATEGORY_PATTERNS = {
    "summarize": re.compile(r"\b(summary|summarize|recap|tldr|tl;dr|3 bullets|bullet)\b", re.IGNORECASE),
    "compare": re.compile(r"\b(compare|match|fit|against|versus|vs\.?)\b", re.IGNORECASE),
    "evidence": re.compile(r"\b(evidence|source|sources|proof|strongest|where|show)\b", re.IGNORECASE),
    "recruiter": re.compile(r"\b(recruiter|hiring|hr|screening|resume version)\b", re.IGNORECASE),
    "job_requirements": re.compile(r"\b(job requirement|job requirements|jd|role requirement|current roles?)\b", re.IGNORECASE),
    "web_context": re.compile(r"\b(web context|latest context|latest web|current context|check latest|look up|search web)\b", re.IGNORECASE),
    "ai_trend": re.compile(r"\b(ai trend|ai trends|agentic|automation trend|trend matters most|developers?\s+right now)\b", re.IGNORECASE),
    "importance": re.compile(r"\b(why|importance|important|matter|matters|impact|relevant)\b", re.IGNORECASE),
    "shorten": re.compile(r"\b(shorten|simplify|simpler|short version|concise|brief)\b", re.IGNORECASE),
    "expand": re.compile(r"\b(expand|elaborate|tell me more|more detail|deep dive|explain)\b", re.IGNORECASE),
    "resume_area": re.compile(
        r"\b(project|projects|skill|skills|education|college|degree|experience|internship|portfolio)\b",
        re.IGNORECASE,
    ),
}

FOLLOW_UP_REFERENCE_RE = re.compile(
    r"\b(this|that|these|those|it|its|above|same|previous|them|there|first|second|one)\b",
    re.IGNORECASE,
)

QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:what\s+(?:are|is|were|was)|which\s+(?:are|is)|who|where|when|why|how|what|which|can you|could you|please|show|tell me|give me|list|explain|compare)\b\s*",
    re.IGNORECASE,
)


def get_last_turn_context(session_id: str) -> Optional[Dict[str, Any]]:
    with SESSION_LOCK:
        existing_session = SESSION_STORE.get(session_id) or {}
        previous_turn = existing_session.get("last_turn")

    return previous_turn if isinstance(previous_turn, dict) else None


def get_previous_user_message(messages: List[MessageItem]) -> str:
    seen_latest_user = False
    for message in reversed(messages):
        if message.role != "user":
            continue
        if not seen_latest_user:
            seen_latest_user = True
            continue
        return message.content
    return ""


def get_previous_assistant_message(messages: List[MessageItem]) -> str:
    for message in reversed(messages[:-1]):
        if message.role == "assistant":
            return message.content
    return ""


def infer_previous_route(previous_turn: Optional[Dict[str, Any]]) -> Optional[str]:
    if not previous_turn:
        return None

    route = previous_turn.get("route")
    if isinstance(route, str) and route:
        return route

    tools_used = previous_turn.get("tools_used")
    tool_names = {
        tool.get("name")
        for tool in tools_used
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    } if isinstance(tools_used, list) else set()

    has_resume = "run_resume_agent" in tool_names
    has_web = "run_web_agent" in tool_names
    if has_resume and has_web:
        return "hybrid"
    if has_resume:
        return "resume"
    if has_web:
        return "web"
    return None


def detect_follow_up_categories(message: str) -> List[str]:
    return [
        category
        for category, pattern in FOLLOW_UP_CATEGORY_PATTERNS.items()
        if pattern.search(message)
    ]


def choose_followup_intent(categories: List[str], message: str) -> Optional[str]:
    priority = [
        "ai_trend",
        "web_context",
        "job_requirements",
        "recruiter",
        "evidence",
        "importance",
        "shorten",
        "expand",
        "summarize",
        "compare",
        "resume_area",
    ]
    category_set = set(categories)
    for category in priority:
        if category in category_set:
            return category
    return None


def is_contextual_follow_up(message: str, previous_user: str, previous_assistant: str) -> bool:
    normalized = compact_whitespace(message)
    if not normalized or not previous_user:
        return False
    if classify_small_talk(normalized):
        return False

    categories = detect_follow_up_categories(normalized)
    word_count = len(re.findall(r"\w+", normalized))
    has_reference = bool(FOLLOW_UP_REFERENCE_RE.search(normalized))
    is_short_command = word_count <= 8 and bool(categories)
    is_question_fragment = word_count <= 12 and normalized.lower().startswith(
        ("why", "how", "what about", "and ", "also ", "then ", "so ")
    )

    return bool(previous_assistant and (has_reference or is_short_command or is_question_fragment))


def normalize_previous_topic(previous_user: str, previous_assistant: str, previous_route: Optional[str]) -> str:
    topic = compact_whitespace(previous_user)
    topic = re.sub(r"[?.!]+$", "", topic)
    topic = QUESTION_PREFIX_RE.sub("", topic).strip()
    topic = re.sub(r"^\b(me|about|the|a|an)\b\s+", "", topic, flags=re.IGNORECASE).strip()

    if not topic and previous_assistant:
        points = split_answer_points(previous_assistant)
        topic = points[0] if points else previous_assistant

    topic = first_complete_sentence(topic, limit=140).rstrip(".")
    lowered = topic.lower()
    if previous_route in {"resume", "hybrid"} and "adarsh" not in lowered:
        topic = f"Adarsh's {topic}" if topic else "Adarsh's resume profile"
    if previous_route == "web" and not topic:
        topic = "the previous web research topic"

    return topic or "the previous topic"


def expand_resume_topic(topic: str, previous_assistant: str) -> str:
    combined = f"{topic} {previous_assistant}".lower()
    if "skill" in combined or "backend" in combined or "spring" in combined:
        return "Adarsh's backend, microservices, Java, Spring Boot, DevOps, Docker, cloud, and AI-related skills"
    if "project" in combined or "portfolio" in combined:
        return "Adarsh's project work, backend implementation, and AI-related portfolio evidence"
    if "background" in combined or "profile" in combined or "candidate" in combined:
        return "Adarsh Kumar's backend/full-stack engineering background"
    return topic


def rewrite_contextual_follow_up(
    message: str,
    *,
    previous_user: str,
    previous_assistant: str,
    previous_route: Optional[str],
) -> str:
    categories = set(detect_follow_up_categories(message))
    topic = normalize_previous_topic(previous_user, previous_assistant, previous_route)
    expanded_topic = expand_resume_topic(topic, previous_assistant)
    lowered = message.lower()

    if "ai_trend" in categories:
        return "Which current AI trend matters most for software developers right now?"
    if "web_context" in categories:
        if previous_route in {"resume", "hybrid"}:
            return f"Check the latest job-market and technology context for {expanded_topic}."
        return f"Check the latest web context for {topic}."
    if "job_requirements" in categories or ("compare" in categories and "current" in lowered):
        return f"Compare {expanded_topic} with current backend, software, and AI engineer job requirements."
    if "recruiter" in categories:
        return f"Rewrite {topic} as a concise recruiter-facing version."
    if "evidence" in categories:
        return f"Show the strongest resume evidence for {topic}."
    if "importance" in categories:
        if "ai" in lowered or previous_route == "web":
            return "Why do current AI trends matter for software developers right now?"
        return f"Explain why {topic} matters and how important it is."
    if "shorten" in categories:
        return f"Shorten and simplify the previous answer about {topic}."
    if "expand" in categories:
        return f"Expand on {topic} with more useful detail."
    if "summarize" in categories:
        return f"Summarize {topic} in a concise, useful way."
    if "compare" in categories:
        return f"Compare {topic} using the previous context."
    if "resume_area" in categories:
        return f"Answer this follow-up about {topic}: {message}"

    return f"Answer this follow-up about {topic}: {message}"


def analyze_contextual_follow_up(
    user_message: str,
    visible_messages: List[MessageItem],
    previous_turn: Optional[Dict[str, Any]] = None,
) -> FollowUpResolution:
    previous_user = ""
    previous_assistant = ""
    previous_route = infer_previous_route(previous_turn)

    if previous_turn:
        previous_user = str(previous_turn.get("user_message") or "")
        previous_assistant = str(previous_turn.get("assistant_message") or "")

    previous_user = previous_user or get_previous_user_message(visible_messages)
    previous_assistant = previous_assistant or get_previous_assistant_message(visible_messages)
    previous_topic = normalize_previous_topic(previous_user, previous_assistant, previous_route) if previous_user else None
    categories = detect_follow_up_categories(user_message)
    followup_intent = choose_followup_intent(categories, user_message)

    if not is_contextual_follow_up(user_message, previous_user, previous_assistant):
        return FollowUpResolution(
            original_user_message=user_message,
            resolved_user_message=user_message,
            followup_intent=followup_intent,
            previous_topic=previous_topic,
        )

    resolved_user_message = rewrite_contextual_follow_up(
        user_message,
        previous_user=previous_user,
        previous_assistant=previous_assistant,
        previous_route=previous_route,
    )
    return FollowUpResolution(
        original_user_message=user_message,
        resolved_user_message=resolved_user_message,
        followup_intent=followup_intent,
        previous_topic=previous_topic,
    )


def resolve_contextual_follow_up(
    user_message: str,
    visible_messages: List[MessageItem],
    previous_turn: Optional[Dict[str, Any]] = None,
) -> str:
    return analyze_contextual_follow_up(user_message, visible_messages, previous_turn).resolved_user_message


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


def store_session_messages(
    session_id: str,
    messages: List[MessageItem],
    last_turn: Optional[Dict[str, Any]] = None,
) -> None:
    with SESSION_LOCK:
        existing_session = SESSION_STORE.get(session_id) or {}
        SESSION_STORE[session_id] = {
            "messages": messages,
            "updated_at": time.time(),
            "last_turn": last_turn if last_turn is not None else existing_session.get("last_turn"),
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
    session: Optional[ClientSession],
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
    last_user_message = next(
        (message.content for message in reversed(visible_messages) if message.role == "user"),
        "",
    )
    last_turn = {
        "user_message": last_user_message,
        "assistant_message": answer,
        "tools_used": [tool.model_dump() for tool in tools_used],
        "route": (extra_log_fields or {}).get("graph_route"),
    }
    store_session_messages(session_id, [*visible_messages, assistant_message], last_turn=last_turn)
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
    session: Optional[ClientSession],
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


async def run_master_agent(request: MasterChatRequest, session: Optional[ClientSession]) -> MasterChatResponse:
    telemetry = TelemetryState(started_at=time.perf_counter())
    session_id = get_or_create_session_id(request.session_id)
    visible_messages = hydrate_visible_messages(session_id, request.messages)
    warnings: List[str] = []
    final_message_id = str(uuid.uuid4())
    user_message = visible_messages[-1].content
    previous_turn = get_last_turn_context(session_id)
    followup_resolution = analyze_contextual_follow_up(
        user_message,
        visible_messages,
        previous_turn,
    )
    resolved_user_message = followup_resolution.resolved_user_message

    async def synthesize_graph_contexts(
        question: str,
        resume_context: Optional[str],
        web_context: Optional[str],
        tool_calls: List[str],
    ) -> str:
        try:
            response = get_groq_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Synthesize the resume and live web specialist outputs into one concise, "
                            "grounded answer. Do not invent facts. If one source is insufficient, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Tools used: {', '.join(tool_calls)}\n\n"
                            f"Resume specialist output:\n{resume_context or 'Not used.'}\n\n"
                            f"Live web specialist output:\n{web_context or 'Not used.'}"
                        ),
                    },
                ],
            )
            prompt_tokens, completion_tokens = extract_groq_usage(response)
            telemetry.add_usage(prompt_tokens, completion_tokens)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            warnings.append(f"Graph final synthesis failed: {exc}")
            return (
                f"Resume context:\n{resume_context or 'No resume context available.'}\n\n"
                f"Live web context:\n{web_context or 'No live web context available.'}"
            )

    initial_state = {
        "user_message": resolved_user_message,
        "chat_history": [message.model_dump() for message in visible_messages],
        "route": None,
        "resume_context": None,
        "web_context": None,
        "final_answer": None,
        "needs_resume": False,
        "needs_web": False,
        "needs_more_info": False,
        "tool_calls": [],
        "retry_count": 0,
        "error": None,
        "session": session,
        "telemetry": telemetry,
        "resume_agent": run_resume_agent,
        "web_agent": run_web_agent,
        "synthesizer": synthesize_graph_contexts,
        "tool_payloads": [],
    }

    try:
        result = await agent_graph.ainvoke(initial_state)
    except Exception as exc:
        answer = f"{ESCALATION_PREFIX} I could not complete the request because the graph failed."
        warnings.append(f"LangGraph failed: {exc}")
        total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
        return finalize_master_chat_response(
            session_id=session_id,
            final_message_id=final_message_id,
            visible_messages=visible_messages,
            answer=answer,
            tools_used=[],
            iterations=0,
            warnings=warnings,
            escalated=True,
            telemetry=telemetry.build_payload(total_latency_ms),
            event_name="master_chat_failed",
            extra_log_fields={"error": str(exc)},
        )

    tools_used: List[ToolUsageRecord] = []
    for payload in result.get("tool_payloads", []):
        tool_name = getattr(payload, "tool_name", "")
        if tool_name in TOOL_REGISTRY:
            tools_used.append(make_tool_usage_record(tool_name, payload))
        if not getattr(payload, "success", False) and getattr(payload, "error", None):
            warning = f"{tool_name}: {payload.error.message}"
            if warning not in warnings:
                warnings.append(warning)

    if result.get("error"):
        warning = str(result["error"])
        if warning not in warnings and not any(warning in existing_warning for existing_warning in warnings):
            warnings.append(warning)

    answer = (result.get("final_answer") or "").strip()
    if not answer:
        answer = f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
        warnings.append("LangGraph returned an empty final answer.")

    structured_response = make_supervisor_response(
        answer,
        user_query=resolved_user_message,
        route=result.get("route"),
        resume_context=result.get("resume_context"),
        web_context=result.get("web_context"),
    )
    escalated = structured_response.final_answer.startswith(ESCALATION_PREFIX)
    total_latency_ms = (time.perf_counter() - telemetry.started_at) * 1000
    event_name = "master_chat_small_talk" if not tools_used else "master_chat_completed"

    return finalize_master_chat_response(
        session_id=session_id,
        final_message_id=final_message_id,
        visible_messages=visible_messages,
        answer=structured_response.final_answer,
        tools_used=tools_used,
        iterations=max(1, len(result.get("tool_calls", []))),
        warnings=warnings,
        escalated=escalated,
        structured_response=structured_response,
        telemetry=telemetry.build_payload(total_latency_ms),
        event_name=event_name,
        extra_log_fields={
            "graph_route": result.get("route"),
            "graph_tool_calls": result.get("tool_calls", []),
            "original_user_message": followup_resolution.original_user_message,
            "resolved_user_message": resolved_user_message,
            "followup_intent": followup_resolution.followup_intent,
            "previous_topic": followup_resolution.previous_topic,
        },
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
    session: Optional[ClientSession] = getattr(http_request.app.state, "mcp_session", None)
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
    session: Optional[ClientSession] = getattr(http_request.app.state, "mcp_session", None)
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


@app.get("/health/mcp")
def health_mcp(http_request: Request):
    return {
        "status": getattr(http_request.app.state, "mcp_status", "unknown"),
        "ready": getattr(http_request.app.state, "mcp_session", None) is not None,
        "required": MCP_REQUIRED,
        "timeout_seconds": MCP_STARTUP_TIMEOUT_SECONDS,
        "server_command": getattr(http_request.app.state, "mcp_server_command", [server_params.command, *server_params.args]),
        "env": getattr(http_request.app.state, "mcp_env_status", get_env_status()),
        "last_error": getattr(http_request.app.state, "mcp_error", None),
        "last_error_details": getattr(http_request.app.state, "mcp_error_details", None),
    }


@app.post("/master-chat", response_model=MasterChatResponse)
async def master_chat(http_request: Request, request: MasterChatRequest) -> MasterChatResponse:
    session: Optional[ClientSession] = getattr(http_request.app.state, "mcp_session", None)
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
