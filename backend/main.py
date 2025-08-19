import os
import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load env (.env with GROQ_API_KEY and TAVILY_API_KEY)
load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_WEB = os.environ.get("USE_WEB_RESEARCH", "true").lower() == "true"

# ---------- State ----------

class OrchestratorState(TypedDict, total=False):
    task: str
    plan: Dict[str, Any]
    research: Dict[str, Any]
    report_markdown: str
    logs: List[str]
    _llm: Any
    web_sources: List[Dict[str, Any]]  # for citations

# ---------- Utils ----------

def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, groq_api_key=GROQ_API_KEY)

def call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def append_log(state: OrchestratorState, line: str) -> None:
    state.setdefault("logs", []).append(line)

def safe_call_planner(llm, system, user, retries=2):
    for _ in range(retries):
        out = call_llm(llm, system, user)
        data = try_parse_json(out)
        if data and all(k in data for k in ["timeline", "workstreams", "risks", "metrics"]):
            return data
        # tighten instructions for retry
        user += "\n\nREMINDER: You must include all required fields with at least 3+ items each."
    return None

# ---------- Nodes ----------

def planner_node(state: OrchestratorState) -> OrchestratorState:
    llm = state["_llm"]
    append_log(state, "Planner: creating detailed timeline, workstreams, and risks.")

    system = (
        "You are the Planner Agent. Your job is to produce **comprehensive project plans**.\n"
        "Every plan MUST include:\n"
        "- At least 3–5 workstreams (Research, Implementation, Validation, Communication, etc.).\n"
        "- Each workstream must have multiple tasks, an owner role, and dependencies.\n"
        "- A timeline with at least 4–6 milestones spread logically (e.g. by week or phase).\n"
        "- At least 3 assumptions.\n"
        "- At least 4 risks with impact + mitigation.\n"
        "- At least 3 metrics of success.\n"
        "\nOutput valid JSON ONLY. Do not include commentary."
    )

    user = f"""
TASK: {state['task']}

Return JSON with this shape (all fields required):

{{
  "objective": "string",
  "assumptions": ["string", "string", ...],
  "timeline": [
    {{
      "phase": "string",
      "milestones": ["string", "string"],
      "deliverables": ["string", "string"]
    }}
  ],
  "workstreams": [
    {{
      "name": "string",
      "tasks": ["string", "string"],
      "owner": "Role",
      "dependencies": ["string", "string"]
    }}
  ],
  "risks": [
    {{
      "risk": "string",
      "impact": "low|medium|high",
      "mitigation": "string"
    }}
  ],
  "metrics": ["string", "string"]
}}
"""
    out = call_llm(llm, system, user)
    data = safe_call_planner(llm, system, user) 


    # Fallback: safe defaults if parsing fails
    if not data:
        data = {
            "objective": state["task"],
            "assumptions": ["TBD"],
            "timeline": [],
            "workstreams": [],
            "risks": [],
            "metrics": []
        }

    append_log(state, "Planner: detailed plan drafted.")
    return {"plan": data}


# Import the web researcher
from agent.research_web import researcher_web_node

def researcher_node(state: OrchestratorState) -> OrchestratorState:
    if USE_WEB:
        append_log(state, "WebResearcher: searching the internet and compiling grounded findings.")
        return researcher_web_node(state)
    else:
        append_log(state, "Researcher: (non-web) compiling generic findings.")
        # fallback to non-web (optional)
        return {"research": {"resources": [], "estimates": [], "validation_checklists": [], "open_questions": []}}

def reporter_node(state: OrchestratorState) -> OrchestratorState:
    llm = state["_llm"]
    append_log(state, "Reporter: compiling final report with citations (Markdown).")

    # Build a simple citation list
    citations_md = ""
    if state.get("web_sources"):
        citations_md = "\n\n**Sources**\n" + "\n".join(
            [f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(state["web_sources"])]
        )

    system = (
        "You are the Reporter Agent. Merge the plan and research into a polished, executive-ready Markdown report. "
        "Include sections: Overview, Assumptions, Timeline (table), Workstreams, Risks & Mitigations, "
        "Resources & Tools, Estimates, Validation Checklists, Open Questions, Next Steps, and a Sources section with citations."
    )
    user = f"""
PLAN:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

RESEARCH:
{json.dumps(state.get('research', {}), ensure_ascii=False, indent=2)}

SOURCES (append at end as a list with [n] labels):
{citations_md}
"""
    md = call_llm(llm, system, user)
    if not md.strip().startswith("#"):
        md = "# Project Plan\n\n" + md
    append_log(state, "Reporter: report assembled.")
    return {"report_markdown": md}

# ---------- Graph ----------

def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reporter", reporter_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()

def run_orchestrator(task: str) -> OrchestratorState:
    app = build_graph()
    llm = get_llm()
    initial: OrchestratorState = {"task": task, "logs": [], "_llm": llm}
    return app.invoke(initial)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    final_state = run_orchestrator(args.task)

    print("\n===== LOGS =====")
    for line in final_state.get("logs", []):
        print("•", line)

    print("\n===== PLAN (JSON) =====")
    print(json.dumps(final_state.get("plan", {}), indent=2, ensure_ascii=False))

    print("\n===== RESEARCH (JSON) =====")
    print(json.dumps(final_state.get("research", {}), indent=2, ensure_ascii=False))

    print("\n===== REPORT (Markdown) =====")
    print(final_state.get("report_markdown", ""))

    # Optional: show which URLs were used
    sources = final_state.get("web_sources", [])
    if sources:
        print("\n===== SOURCES USED =====")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] {s['title']} — {s['url']}")
