import json
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from agent.planner_agent import planner_node
from agent.reviewer_agent import reviewer_node
from retrieval.research_web import researcher_web_node

# ----- Shared State -----
class OrchestratorState(TypedDict, total=False):
    task: str
    plan: Dict[str, Any]
    research: Dict[str, Any]
    report_markdown: str
    logs: List[str]
    _llm: Any
    web_sources: List[Dict[str, Any]]
    # internal control flags/counters
    validated: bool
    review_attempts: int
    max_review_attempts: int

def _append_log(state: OrchestratorState, line: str) -> None:
    state.setdefault("logs", []).append(line)

def _call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

# ----- Lightweight plan validator (no-LLM, prevents infinite loops) -----
def _plan_issues(plan: Dict[str, Any], task: str) -> List[str]:
    issues: List[str] = []
    if not isinstance(plan, dict) or not plan:
        return ["plan_missing_or_not_dict"]

    # Required top-level sections
    if not plan.get("assumptions") or len(plan["assumptions"]) < 3:
        issues.append("assumptions>=3")

    # Timeline checks
    timeline = plan.get("timeline") or []
    if len(timeline) < 4:
        issues.append("timeline>=4")
    else:
        # each phase must have milestones[] and deliverables[]
        for idx, phase in enumerate(timeline, start=1):
            if not isinstance(phase, dict):
                issues.append(f"timeline_phase_{idx}_not_dict")
                continue
            if not phase.get("milestones") or not isinstance(phase["milestones"], list):
                issues.append(f"timeline_phase_{idx}_milestones_missing")
            if not phase.get("deliverables") or not isinstance(phase["deliverables"], list):
                issues.append(f"timeline_phase_{idx}_deliverables_missing")

    # If task looks like itinerary/day/week, require more granular timeline
    lowered = (task or "").lower()
    if any(w in lowered for w in ["day", "week", "itinerary"]):
        # require at least 7 entries to discourage oversimplified timelines
        if len(timeline) < 7:
            issues.append("timeline_daily_breakdown_required")

    # Workstreams
    workstreams = plan.get("workstreams") or []
    if len(workstreams) < 4:
        issues.append("workstreams>=4")
    else:
        names_seen = set()
        for i, ws in enumerate(workstreams, start=1):
            if not isinstance(ws, dict):
                issues.append(f"workstream_{i}_not_dict")
                continue
            if not ws.get("tasks") or not isinstance(ws["tasks"], list):
                issues.append(f"workstream_{i}_tasks_missing")
            if not ws.get("owner"):
                issues.append(f"workstream_{i}_owner_missing")
            if not ws.get("dependencies") or not isinstance(ws["dependencies"], list):
                issues.append(f"workstream_{i}_dependencies_missing")
            name = (ws.get("name") or "").strip().lower()
            if name:
                if name in names_seen:
                    issues.append("workstreams_unique")
                names_seen.add(name)

    # Risks
    risks = plan.get("risks") or []
    if len(risks) < 4:
        issues.append("risks>=4")
    else:
        seen = set()
        for i, r in enumerate(risks, start=1):
            if not isinstance(r, dict):
                issues.append(f"risk_{i}_not_dict")
                continue
            risk_name = (r.get("risk") or "").strip().lower()
            if not risk_name:
                issues.append(f"risk_{i}_name_missing")
            if risk_name in seen:
                issues.append("risks_diverse")
            seen.add(risk_name)
            if not r.get("impact"):
                issues.append(f"risk_{i}_impact_missing")
            if not r.get("mitigation"):
                issues.append(f"risk_{i}_mitigation_missing")

    # Metrics
    if not plan.get("metrics") or len(plan["metrics"]) < 3:
        issues.append("metrics>=3")

    return issues

def validator_node(state: OrchestratorState) -> OrchestratorState:
    """Pure-Python gate to stop infinite planner<->reviewer loops."""
    plan = state.get("plan") or {}
    task = state.get("task", "")
    issues = _plan_issues(plan, task)

    if issues:
        attempts = int(state.get("review_attempts", 0)) + 1
        max_attempts = int(state.get("max_review_attempts", 3))
        _append_log(
            state,
            f"Validator: plan still has issues -> {', '.join(issues)} (attempt {attempts}/{max_attempts})."
        )
        return {"validated": False, "review_attempts": attempts}
    else:
        _append_log(state, "Validator: plan passed structural checks.")
        return {"validated": True}

# ----- Reporter -----
def reporter_node(state: OrchestratorState) -> OrchestratorState:
    llm: ChatGroq = state["_llm"]
    _append_log(state, "Reporter: compiling final report with citations (Markdown).")

    citations_md = ""
    if state.get("web_sources"):
        citations_md = "\n\n**Sources**\n" + "\n".join(
            [f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(state["web_sources"])]
        )

    system = (
        "You are the Reporter Agent. Merge the plan and research into a polished, executive-ready Markdown report. "
        "Include sections: Overview, Assumptions, Timeline (table), Workstreams, Risks & Mitigations, "
        "Resources & Tools, Estimates, Validation Checklists, Open Questions, Next Steps, and a Sources section with citations. "
        "Do not invent facts; when unsure, keep it generic."
    )
    user = f"""
PLAN:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

RESEARCH:
{json.dumps(state.get('research', {}), ensure_ascii=False, indent=2)}

SOURCES (append at end as a list with [n] labels):
{citations_md}
"""
    md = _call_llm(llm, system, user)
    if not md.strip().startswith("#"):
        md = "# Project Plan\n\n" + md
    _append_log(state, "Reporter: report assembled.")
    return {"report_markdown": md}

# ----- Graph Builder -----
def build_graph(use_web: bool = True):
    graph = StateGraph(OrchestratorState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("validator", validator_node)
    graph.add_node("researcher", researcher_web_node if use_web else (lambda s: {"research": {}}))
    graph.add_node("reporter", reporter_node)

    # Entry point
    graph.set_entry_point("planner")

    # Linear edges with a gated loop
    graph.add_edge("planner", "reviewer")
    graph.add_edge("reviewer", "validator")

    # Validator decides whether to loop or move forward
    def validation_router(state: OrchestratorState) -> str:
        if state.get("validated", False):
            return "researcher"
        # Hard stop if we hit the retry limit
        if int(state.get("review_attempts", 0)) >= int(state.get("max_review_attempts", 3)):
            return "researcher"
        # Otherwise loop back to planner
        return "planner"

    graph.add_conditional_edges(
        "validator",
        validation_router,
        {"planner": "planner", "researcher": "researcher"}
    )

    graph.add_edge("researcher", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()

def run_orchestrator(task: str, llm: ChatGroq, use_web: bool = True) -> OrchestratorState:
    app = build_graph(use_web=use_web)
    initial: OrchestratorState = {
        "task": task,
        "logs": [],
        "_llm": llm,
        "validated": False,
        "review_attempts": 0,
        "max_review_attempts": 3,  # change if you want more/less refinement loops
    }
    # Add a reasonable recursion limit to avoid long traces even if misconfigured
    return app.invoke(initial, config={"recursion_limit": 20})
