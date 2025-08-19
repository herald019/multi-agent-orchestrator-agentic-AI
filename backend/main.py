import os
import json
from typing import Any, Dict, List, TypedDict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load env (.env with GROQ_API_KEY and TAVILY_API_KEY)
load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_WEB = os.environ.get("USE_WEB_RESEARCH", "true").lower() == "true"


class OrchestratorState(TypedDict, total=False):
    task: str
    plan: Dict[str, Any]
    research: Dict[str, Any]
    report_markdown: str
    logs: List[str]
    _llm: Any
    web_sources: List[Dict[str, Any]]  # for citations


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    # Low temperature for structure + accuracy
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, groq_api_key=GROQ_API_KEY)


if __name__ == "__main__":
    from agent.orchestrator import run_orchestrator

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    llm = get_llm()
    final_state: OrchestratorState = run_orchestrator(args.task, llm=llm, use_web=USE_WEB)

    print("\n===== LOGS =====")
    for line in final_state.get("logs", []):
        print("•", line)

    print("\n===== PLAN (JSON) =====")
    print(json.dumps(final_state.get("plan", {}), indent=2, ensure_ascii=False))

    print("\n===== RESEARCH (JSON) =====")
    print(json.dumps(final_state.get("research", {}), indent=2, ensure_ascii=False))

    print("\n===== REPORT (Markdown) =====")
    print(final_state.get("report_markdown", ""))

    sources = final_state.get("web_sources", [])
    if sources:
        print("\n===== SOURCES USED =====")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] {s['title']} — {s['url']}")
