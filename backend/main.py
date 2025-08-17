import os
import json
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

# ---------- Config ----------

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # must be set


# ---------- State Definition ----------

class OrchestratorState(TypedDict, total=False):
    task: str                                    # user input, e.g., "Plan a hackathon in 2 weeks"
    plan: Dict[str, Any]                          # structured plan (milestones, workstreams, risks)
    research: Dict[str, Any]                      # findings from researcher
    report_markdown: str                          # final report in markdown
    logs: List[str]                               # running log for UI/debug


# ---------- Utilities ----------

def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    # Temperature kept low to encourage consistent JSON
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, groq_api_key=GROQ_API_KEY)

def call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    # Extract first {...} block to be forgiving with model outputs
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def append_log(state: OrchestratorState, line: str) -> None:
    state.setdefault("logs", []).append(line)


# ---------- Agent Nodes ----------

def planner_node(state: OrchestratorState) -> OrchestratorState:
    llm = get_llm()
    append_log(state, "Planner: creating timeline, workstreams, and risks.")

    system = (
        "You are the Planner Agent. You create crisp, actionable project plans with milestones, "
        "workstreams, deliverables, dependencies, and risks. Output valid JSON ONLY."
    )
    user = f"""
Create a structured plan for the following task. Keep it realistic and time-bounded.

TASK: {state['task']}

Return JSON with this exact shape:
{{
  "objective": "...",
  "assumptions": ["...", "..."],
  "timeline": [
    {{"week": "W1", "milestones": ["..."], "deliverables": ["..."]}},
    {{"week": "W2", "milestones": ["..."], "deliverables": ["..."]}}
  ],
  "workstreams": [
    {{"name": "Research", "tasks": ["..."], "owner": "Role", "dependencies": []}},
    {{"name": "Design", "tasks": ["..."], "owner": "Role", "dependencies": ["Research"]}}
  ],
  "risks": [
    {{"risk": "...", "impact": "low|medium|high", "mitigation": "..."}}
  ],
  "metrics": ["...", "..."]
}}
Only JSON. No extra commentary.
"""
    out = call_llm(llm, system, user)
    data = try_parse_json(out)
    if data is None:
        append_log(state, "Planner: model did not return valid JSON; falling back to minimal skeleton.")
        data = {
            "objective": state["task"],
            "assumptions": [],
            "timeline": [],
            "workstreams": [],
            "risks": [],
            "metrics": []
        }
    append_log(state, "Planner: plan drafted.")
    return {"plan": data}


def researcher_node(state: OrchestratorState) -> OrchestratorState:
    llm = get_llm()
    append_log(state, "Researcher: analyzing plan and generating supporting findings & open questions.")

    system = (
        "You are the Research Agent. Given a plan, identify unknowns, propose realistic options, "
        "and enrich with estimates and checklists. Do NOT browse the web; reason from first principles. "
        "Return valid JSON ONLY."
    )
    plan_json = json.dumps(state["plan"], ensure_ascii=False, indent=2)
    user = f"""
Given the plan below, produce supporting research for each workstream:
- key resources/tools you'll likely need,
- rough time/cost estimates (relative),
- a checklist to validate completion,
- top 5 questions to clarify with stakeholders.

PLAN:
{plan_json}

Return JSON with:
{{
  "resources": [{{"workstream": "...", "tools": ["..."], "templates": ["..."]}} ],
  "estimates": [{{"workstream": "...", "effort": "S/M/L", "notes": "..."}} ],
  "validation_checklists": [{{"workstream": "...", "checklist": ["..."]}} ],
  "open_questions": ["...", "...", "..."]
}}
Only JSON. No extra commentary.
"""
    out = call_llm(llm, system, user)
    data = try_parse_json(out)
    if data is None:
        append_log(state, "Researcher: model did not return valid JSON; using empty research.")
        data = {
            "resources": [],
            "estimates": [],
            "validation_checklists": [],
            "open_questions": []
        }
    append_log(state, "Researcher: research compiled.")
    return {"research": data}


def reporter_node(state: OrchestratorState) -> OrchestratorState:
    llm = get_llm()
    append_log(state, "Reporter: compiling final report in markdown.")

    system = (
        "You are the Reporter Agent. Merge a plan and research into a polished, executive-ready "
        "Markdown report. Be concise, actionable, and well-structured."
    )
    user = f"""
Create a polished Markdown report that merges the PLAN and RESEARCH below.
Include sections: Overview, Assumptions, Timeline (table), Workstreams (tasks & owners),
Risks & Mitigations, Resources & Tools, Estimates, Validation Checklists, Open Questions, and Next Steps.

PLAN JSON:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

RESEARCH JSON:
{json.dumps(state['research'], ensure_ascii=False, indent=2)}
"""
    md = call_llm(llm, system, user)
    # Minimal sanity check
    if not md.strip().startswith("#"):
        md = "# Project Plan\n\n" + md
    append_log(state, "Reporter: report assembled.")
    return {"report_markdown": md}


# ---------- Graph Wiring ----------

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


# ---------- Public API ----------

def run_orchestrator(task: str) -> OrchestratorState:
    """
    Runs the multi-agent pipeline for a single task prompt.
    Returns the final state, including: plan, research, report_markdown, logs.
    """
    app = build_graph()
    initial: OrchestratorState = {"task": task, "logs": []}
    result: OrchestratorState = app.invoke(initial)  # synchronous single pass
    return result


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Task Orchestrator (Groq + LangGraph)")
    parser.add_argument("--task", type=str, required=True, help="Project request, e.g., 'Plan a hackathon in 2 weeks'")
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
