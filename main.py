import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tavily import TavilyClient

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)
MAX_AGENT_ITERATIONS = 5
DEFAULT_WEB_RESULTS = 3
DEFAULT_RESUME_RESULTS = 3
ESCALATION_PREFIX = "Escalation:"
SMALL_TALK_MAX_LENGTH = 120
MAX_TOOL_RESULT_CHARS = 2500
AGENT_REQUEST_RETRIES = 2
AGENT_RETRY_DELAY_SECONDS = 2
AGENT_LOOP_DELAY_SECONDS = 1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

client = Groq(api_key=groq_api_key) if groq_api_key else None
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

SESSION_STORE: Dict[str, Dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BulletproofHFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not configured.")

        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        last_error = "Unknown error"

        for attempt in range(3):
            print("Sending request to Hugging Face...")
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=60)

            try:
                result = response.json()
            except ValueError:
                result = None

            if isinstance(result, list):
                return result

            if isinstance(result, dict):
                last_error = result.get("error") or result.get("message") or json.dumps(result)
            else:
                last_error = response.text[:500] or f"HTTP {response.status_code}"

            if attempt < 2:
                print(f"HF Error Detected: {last_error}. Retrying in 5 seconds...")
                time.sleep(5)

        raise RuntimeError(f"Hugging Face embeddings request failed: {last_error}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


print("Loading Vector Database...")
embedding_model = BulletproofHFEmbeddings()
db = Chroma(persist_directory="./my_vector_db", embedding_function=embedding_model)


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


class SearchLiveWebInput(StrictBaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="A precise web search query for current or live information.",
    )


class SearchResumeDatabaseInput(StrictBaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description="A semantic search query for finding information in the resume database.",
    )


class ToolErrorPayload(StrictBaseModel):
    type: str
    message: str


class ToolExecutionPayload(StrictBaseModel):
    success: bool
    tool_name: str
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[ToolErrorPayload] = None


class ToolUsageRecord(StrictBaseModel):
    name: str
    label: str
    query: str
    status: Literal["success", "error"]
    result_count: int = 0
    error: Optional[str] = None


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
    tools_used: List[ToolUsageRecord] = Field(default_factory=list)
    iterations: int
    escalated: bool = False
    warnings: List[str] = Field(default_factory=list)


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
    executor: Callable[[StrictBaseModel], ToolExecutionPayload]


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


def search_live_web_executor(tool_input: SearchLiveWebInput) -> ToolExecutionPayload:
    if tavily_client is None:
        return build_tool_error(
            tool_name="search_live_web",
            query=tool_input.query,
            error_type="configuration_error",
            message="TAVILY_API_KEY is not configured.",
        )

    try:
        response = tavily_client.search(
            query=tool_input.query,
            search_depth="basic",
            max_results=DEFAULT_WEB_RESULTS,
        )
        results = [
            {
                "url": result.get("url", ""),
                "content": truncate_text(result.get("content", "")),
            }
            for result in response.get("results", [])
        ]
        return ToolExecutionPayload(
            success=True,
            tool_name="search_live_web",
            query=tool_input.query,
            results=results,
        )
    except Exception as exc:
        return build_tool_error(
            tool_name="search_live_web",
            query=tool_input.query,
            error_type="tool_execution_error",
            message=f"Live web search failed: {exc}",
        )


def search_resume_database_executor(tool_input: SearchResumeDatabaseInput) -> ToolExecutionPayload:
    try:
        docs = db.similarity_search(tool_input.query, k=DEFAULT_RESUME_RESULTS)
        results = []
        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "content": truncate_text(doc.page_content),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                }
            )

        return ToolExecutionPayload(
            success=True,
            tool_name="search_resume_database",
            query=tool_input.query,
            results=results,
        )
    except Exception as exc:
        return build_tool_error(
            tool_name="search_resume_database",
            query=tool_input.query,
            error_type="tool_execution_error",
            message=f"Resume database search failed: {exc}",
        )


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "search_live_web": ToolSpec(
        name="search_live_web",
        label="Searched the web",
        description="Search the live web for current events, recent facts, or information that may have changed.",
        input_model=SearchLiveWebInput,
        executor=search_live_web_executor,
    ),
    "search_resume_database": ToolSpec(
        name="search_resume_database",
        label="Read resume",
        description="Search the local resume database for verified information about the candidate's background, projects, skills, and experience.",
        input_model=SearchResumeDatabaseInput,
        executor=search_resume_database_executor,
    ),
}

TOOLS_MENU = [build_tool_schema(tool_spec) for tool_spec in TOOL_REGISTRY.values()]


def as_groq_messages(messages: List[MessageItem]) -> List[Dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def build_master_system_prompt() -> str:
    return f"""You are a unified production assistant.
Rules:
1. Use search_resume_database for questions about the resume, candidate profile, projects, skills, education, or past experience.
2. Use search_live_web for current events, recent facts, or information that may have changed after training.
3. If the user is just saying hello, goodbye, thanking you, or making small talk, DO NOT use any tools. Reply conversationally.
4. NEVER output raw XML, tool markup, or <function> tags under any circumstances.
5. You may call tools multiple times when needed, but keep each tool call focused.
6. Never invent tool outputs. If a tool fails, either try a different tool or explain the limitation.
7. If you still do not have enough verified information after using the tools, start your answer with '{ESCALATION_PREFIX}' and clearly state what is missing.
8. Give the final answer directly and concisely. Do not narrate your internal chain of thought.
9. Prefer grounded answers over broad speculation.
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


def execute_tool_call(tool_call: Any) -> tuple[ToolExecutionPayload, ToolUsageRecord]:
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

    payload = tool_spec.executor(validated_input)
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
    event_name: str = "master_chat_completed",
    extra_log_fields: Optional[Dict[str, Any]] = None,
) -> MasterChatResponse:
    assistant_message = MessageItem(role="assistant", content=answer)
    store_session_messages(session_id, [*visible_messages, assistant_message])
    log_event(
        event_name,
        {
            "session_id": session_id,
            "message_id": final_message_id,
            "iterations": iterations,
            "tools_used": [tool.model_dump() for tool in tools_used],
            "escalated": escalated,
            **(extra_log_fields or {}),
        },
    )
    return MasterChatResponse(
        session_id=session_id,
        message_id=final_message_id,
        answer=answer,
        tools_used=tools_used,
        iterations=iterations,
        escalated=escalated,
        warnings=warnings,
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


def append_tool_results(
    session_id: str,
    iteration: int,
    llm_messages: List[Dict[str, Any]],
    tool_calls: List[Any],
    tools_used: List[ToolUsageRecord],
    warnings: List[str],
) -> None:
    for tool_call in tool_calls:
        payload, usage_record = execute_tool_call(tool_call)
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


def run_master_agent(request: MasterChatRequest) -> MasterChatResponse:
    session_id = get_or_create_session_id(request.session_id)
    visible_messages = hydrate_visible_messages(session_id, request.messages)
    llm_messages: List[Dict[str, Any]] = as_groq_messages(visible_messages)
    tools_used: List[ToolUsageRecord] = []
    warnings: List[str] = []
    final_message_id = str(uuid.uuid4())
    small_talk_reply = build_small_talk_reply(visible_messages[-1].content)

    if small_talk_reply is not None:
        return finalize_master_chat_response(
            session_id=session_id,
            final_message_id=final_message_id,
            visible_messages=visible_messages,
            answer=small_talk_reply,
            tools_used=tools_used,
            iterations=0,
            warnings=warnings,
            escalated=False,
            event_name="master_chat_small_talk",
        )

    try:
        groq_client = get_groq_client()
    except RuntimeError as exc:
        answer = f"{ESCALATION_PREFIX} {exc}"
        return finalize_master_chat_response(
            session_id=session_id,
            final_message_id=final_message_id,
            visible_messages=visible_messages,
            answer=answer,
            tools_used=tools_used,
            iterations=0,
            warnings=[str(exc)],
            escalated=True,
        )

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        try:
            response = request_agent_step(groq_client, llm_messages)
        except Exception as exc:
            answer = f"{ESCALATION_PREFIX} I could not complete the request because the model call failed."
            warnings.append(f"Groq request failed: {exc}")
            return finalize_master_chat_response(
                session_id=session_id,
                final_message_id=final_message_id,
                visible_messages=visible_messages,
                answer=answer,
                tools_used=tools_used,
                iterations=iteration,
                warnings=warnings,
                escalated=True,
                event_name="master_chat_failed",
                extra_log_fields={"error": str(exc)},
            )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls or []

        if not tool_calls:
            answer = (response_message.content or "").strip()
            if not answer:
                answer = f"{ESCALATION_PREFIX} I do not have enough verified information to answer that safely."
                warnings.append("The model returned an empty final answer.")

            escalated = answer.startswith(ESCALATION_PREFIX)
            return finalize_master_chat_response(
                session_id=session_id,
                final_message_id=final_message_id,
                visible_messages=visible_messages,
                answer=answer,
                tools_used=tools_used,
                iterations=iteration,
                warnings=warnings,
                escalated=escalated,
            )

        llm_messages.append(format_assistant_tool_message(response_message))
        append_tool_results(session_id, iteration, llm_messages, tool_calls, tools_used, warnings)
        time.sleep(AGENT_LOOP_DELAY_SECONDS)

    answer = (
        f"{ESCALATION_PREFIX} I reached the maximum number of tool steps for this request and "
        "do not yet have enough verified information to answer confidently."
    )
    warnings.append("The agent loop hit the iteration limit.")
    return finalize_master_chat_response(
        session_id=session_id,
        final_message_id=final_message_id,
        visible_messages=visible_messages,
        answer=answer,
        tools_used=tools_used,
        iterations=MAX_AGENT_ITERATIONS,
        warnings=warnings,
        escalated=True,
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
def ask_document(request: QuestionRequest):
    tool_result = search_resume_database_executor(SearchResumeDatabaseInput(query=request.question))
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
    }


@app.post("/agent")
def run_agent(request: AgentRequest):
    master_response = run_master_agent(
        MasterChatRequest(messages=[MessageItem(role="user", content=request.question)])
    )

    web_tool = next(
        (tool for tool in master_response.tools_used if tool.name == "search_live_web"),
        None,
    )

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
def master_chat(request: MasterChatRequest) -> MasterChatResponse:
    return run_master_agent(request)


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
