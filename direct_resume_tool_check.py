import json

from mcp_server import search_resume_database


def run_check() -> None:
    raw_payload = search_resume_database("technical skills")
    payload = json.loads(raw_payload)

    assert payload.get("success") is True, payload
    assert payload.get("tool_name") == "search_resume_database", payload
    assert payload.get("results"), payload

    print(
        json.dumps(
            {
                "status": "ok",
                "result_count": len(payload["results"]),
                "top_section": payload["results"][0].get("section_name"),
            },
            default=str,
        )
    )


if __name__ == "__main__":
    run_check()
