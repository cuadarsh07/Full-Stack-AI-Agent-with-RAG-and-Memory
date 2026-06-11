from main import (
    analyze_contextual_follow_up,
    MessageItem,
    build_concise_answer,
    build_dynamic_follow_ups,
    classify_small_talk,
    resolve_contextual_follow_up,
    summarize_resume_results,
)


def assert_short(label: str, answer: str, max_lines: int = 5) -> None:
    lines = [line for line in answer.splitlines() if line.strip()]
    bullet_count = sum(1 for line in lines if line.strip().startswith("- "))
    assert len(lines) <= max_lines, f"{label} is too long: {lines}"
    assert bullet_count <= 4, f"{label} has too many bullets: {bullet_count}"
    assert "..." not in answer, f"{label} contains clipped ellipses: {answer}"
    assert "Based on the provided" not in answer, f"{label} leaked internal phrasing: {answer}"


def run_checks() -> None:
    hi = build_concise_answer("Hi. I can help with Adarsh's resume, projects, skills, or current web research.")
    assert_short("Hi", hi, max_lines=1)

    resume = build_concise_answer(
        "Adarsh has experience with Python, FastAPI, LangGraph, vector databases, and API-backed applications. "
        "His resume also shows backend implementation work and project-oriented development. "
        "This is relevant for software engineering roles.",
        user_query="What are Adarsh's strongest resume points?",
        route="resume",
        resume_context=(
            "Adarsh has Python and FastAPI experience. He has worked with LangGraph routing. "
            "He has vector database and API integration experience."
        ),
    )
    assert_short("Resume answer", resume)

    web = build_concise_answer(
        "AI developer tooling is moving toward agentic workflows, retrieval, and production evaluation. "
        "Teams are also focusing on latency, cost controls, and safer tool use.",
        user_query="latest AI trends for developers",
        route="web",
        web_context=(
            "Agentic coding tools are becoming more common. Retrieval and evaluation are important. "
            "Production teams care about latency and cost."
        ),
    )
    assert_short("Web answer", web)

    hybrid = build_concise_answer(
        "Adarsh's backend and vector database work connects well with current AI developer trends.",
        user_query="Compare Adarsh resume with latest AI developer trends",
        route="hybrid",
        resume_context="Adarsh has Python, FastAPI, vector database, and LangGraph experience.",
        web_context="Current AI developer trends include agentic workflows, RAG, and evaluation.",
    )
    assert_short("Hybrid answer", hybrid, max_lines=4)
    assert "Resume side:" in hybrid
    assert "Current AI/web side:" in hybrid

    weak = build_concise_answer(
        "The resume evidence is insufficient and does not mention named AI projects.",
        user_query="What AI projects are in Adarsh's resume?",
        route="resume",
        resume_context="Closest evidence includes Python, FastAPI, and vector database work.",
    )
    assert weak.startswith("I don't see named AI projects in the resume evidence yet.")
    assert_short("Weak evidence answer", weak, max_lines=3)

    ai_resume_followups = build_dynamic_follow_ups(
        "What AI projects are in Adarsh's resume?",
        route="resume",
        has_resume=True,
    )
    assert ai_resume_followups == [
        "Show Adarsh's strongest AI-related resume points",
        "What AI project should Adarsh add to his portfolio?",
        "Compare Adarsh's AI experience with current AI job requirements",
    ]

    ai_web_followups = build_dynamic_follow_ups(
        "latest AI trends",
        route="web",
        has_web=True,
    )
    assert ai_web_followups == [
        "Summarize the current AI trend context in 3 bullets",
        "Which AI trend matters most for software developers right now?",
        "Compare current AI trends with Adarsh's backend skills",
    ]

    visible_messages = [
        MessageItem(role="user", content="What are Adarsh's strongest technical skills?"),
        MessageItem(
            role="assistant",
            content="Adarsh's strongest skills are Python, FastAPI, LangGraph, and vector databases.",
        ),
        MessageItem(role="user", content="Compare this with current job requirements"),
    ]
    previous_turn = {
        "user_message": "What are Adarsh's strongest technical skills?",
        "assistant_message": "Adarsh's strongest skills are Python, FastAPI, LangGraph, and vector databases.",
        "route": "resume",
        "tools_used": [{"name": "run_resume_agent"}],
    }
    resolved_compare = resolve_contextual_follow_up(
        "Compare this with current job requirements",
        visible_messages,
        previous_turn,
    )
    assert resolved_compare == (
        "Compare Adarsh's backend, microservices, Java, Spring Boot, DevOps, Docker, cloud, "
        "and AI-related skills with current backend, software, and AI engineer job requirements."
    )

    compare_analysis = analyze_contextual_follow_up(
        "Compare this with current job requirements",
        visible_messages,
        previous_turn,
    )
    assert compare_analysis.original_user_message == "Compare this with current job requirements"
    assert compare_analysis.resolved_user_message == resolved_compare
    assert compare_analysis.followup_intent == "job_requirements"
    assert compare_analysis.previous_topic == "Adarsh's strongest technical skills"

    resolved_evidence = resolve_contextual_follow_up(
        "Show the strongest resume evidence",
        [
            *visible_messages[:-1],
            MessageItem(role="user", content="Show the strongest resume evidence"),
        ],
        previous_turn,
    )
    assert resolved_evidence == "Show the strongest resume evidence for Adarsh's strongest technical skills."

    resolved_recruiter = resolve_contextual_follow_up(
        "summarize this for a recruiter",
        [
            *visible_messages[:-1],
            MessageItem(role="user", content="summarize this for a recruiter"),
        ],
        previous_turn,
    )
    assert resolved_recruiter == "Rewrite Adarsh's strongest technical skills as a concise recruiter-facing version."

    resolved_ai_trend = resolve_contextual_follow_up(
        "Which AI trend matters most for developers?",
        [
            *visible_messages[:-1],
            MessageItem(role="user", content="Which AI trend matters most for developers?"),
        ],
        previous_turn,
    )
    assert resolved_ai_trend == "Which current AI trend matters most for software developers right now?"

    resolved_web_context = resolve_contextual_follow_up(
        "Check the latest web context",
        [
            *visible_messages[:-1],
            MessageItem(role="user", content="Check the latest web context"),
        ],
        previous_turn,
    )
    assert resolved_web_context == (
        "Check the latest job-market and technology context for Adarsh's backend, microservices, "
        "Java, Spring Boot, DevOps, Docker, cloud, and AI-related skills."
    )

    casual_message = "how are u bro"
    assert classify_small_talk(casual_message) == "small_talk"
    assert resolve_contextual_follow_up(
        casual_message,
        [
            *visible_messages[:-1],
            MessageItem(role="user", content=casual_message),
        ],
        previous_turn,
    ) == casual_message

    resume_summary = summarize_resume_results(
        "Give me a short overview of Adarsh Kumar's background.",
        [
            {
                "content": "Adarsh Kumar cuadarsh07@gmail.com | github.com/adarshak07 | linkedin.com/in/adarsh07 | phone(+91) 8789473625",
                "metadata": {"section_name": "Profile", "chunk_index": 0},
            },
            {
                "content": "Neeve.ai - Graduate Engineer Trainee Oct 2025 - Present",
                "metadata": {"section_name": "Experience", "chunk_index": 2},
            },
            {
                "content": "Programming Languages: Java, C++, SQL, HTML/CSS Backend & Microservices: Spring Boot, REST APIs, Microservice Design",
                "metadata": {"section_name": "Skills", "chunk_index": 10},
            },
        ],
    )
    assert "cuadarsh07" not in resume_summary
    assert "Experience:" not in resume_summary
    assert "Skills:" not in resume_summary
    assert "His experience includes" in resume_summary
    assert "His strongest technical areas include" in resume_summary


if __name__ == "__main__":
    run_checks()
    print("manual_master_chat_checks passed")
