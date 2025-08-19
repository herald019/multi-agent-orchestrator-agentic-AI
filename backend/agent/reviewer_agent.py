import json
from typing import Any, Dict, List, TypedDict
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

class OrchestratorState(TypedDict, total=False):
    task: str
    plan: Dict[str, Any]
    logs: List[str]
    _llm: Any

def _append_log(state: OrchestratorState, line: str) -> None:
    state.setdefault("logs", []).append(line)

def _call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

def _must_fix(plan: Dict[str, Any], task: str = "") -> List[str]:
    probs = []
    if not plan.get("assumptions") or len(plan["assumptions"]) < 3:
        probs.append("assumptions>=3")
    if not plan.get("timeline") or len(plan["timeline"]) < 4:
        probs.append("timeline>=4")
    if not plan.get("workstreams") or len(plan["workstreams"]) < 4:
        probs.append("workstreams>=4")
    if not plan.get("risks") or len(plan["risks"]) < 4:
        probs.append("risks>=4")
    if not plan.get("metrics") or len(plan["metrics"]) < 3:
        probs.append("metrics>=3")

    # Each workstream must have required fields
    for i, ws in enumerate(plan.get("workstreams", []), 1):
        if not ws.get("tasks") or not ws.get("owner") or not ws.get("dependencies"):
            probs.append(f"workstream_{i}_fields")

    # --- NEW RULES ---
    # Force daily breakdown if "day" or "week" in task
    if any(word in task.lower() for word in ["day", "week", "itinerary"]):
        if len(plan.get("timeline", [])) < 7:
            probs.append("timeline_daily_breakdown_required")

    # Require open questions resolved
    if plan.get("open_questions"):
        probs.append("open_questions_resolved")

    # Risks must be varied
    risks = [r.get("risk", "").lower() for r in plan.get("risks", [])]
    if len(risks) != len(set(risks)):
        probs.append("risks_diverse")

    # Workstreams must be unique
    names = [ws.get("name", "").lower() for ws in plan.get("workstreams", [])]
    if len(names) != len(set(names)):
        probs.append("workstreams_unique")

    return probs


def reviewer_node(state: OrchestratorState) -> OrchestratorState:
    _append_log(state, "Reviewer: validating plan structure and completeness.")
    plan = state.get("plan") or {}
    issues = _must_fix(plan, task=state.get("task", ""))

    if not issues:
        _append_log(state, "Reviewer: plan passed validation.")
        # Mark validated so orchestrator can move forward
        return {"validated": True}

    _append_log(state, f"Reviewer: found issues -> {', '.join(issues)}. Requesting refinement.")

    llm: ChatGroq = state["_llm"]
    system = (
        "You are the Reviewer Agent. You receive a project plan in JSON format "
        "and a list of missing/weak elements. Your job is to refine and correct it "
        "so that it is complete, detailed, realistic, and implementable. "
        "Do not delete good content. Instead, expand, enrich, and ground it in real-world best practices. "
        "If the task involves a timeline (days/weeks/itinerary), ensure a day-by-day breakdown. "
        "If there are open_questions, answer them inside the plan. "
        "Risks should be varied and realistic, with clear mitigations. "
        "Workstreams must be distinct and balanced. "
        "Always return valid JSON only."
    )

    user = f"""
TASK: {state['task']}

PROBLEMS:
{issues}

CURRENT PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

CONSTRAINTS (must all be satisfied):
- >=3 assumptions
- >=4 timeline phases; if the task mentions days/weeks/itinerary, expand into daily breakdowns
- >=4 workstreams; each must have tasks[], owner, dependencies[], and be unique
- >=4 diverse risks (each with impact + mitigation)
- >=3 metrics
- No unresolved open_questions (convert them into answered sections or notes)
Return only the corrected JSON (no commentary).
"""
    out = _call_llm(llm, system, user)

    # Parse corrected JSON
    try:
        start, end = out.find("{"), out.rfind("}")
        fixed = json.loads(out[start:end+1]) if start != -1 and end != -1 else plan
    except Exception:
        fixed = plan

    _append_log(state, "Reviewer: plan corrected.")
    return {"plan": fixed, "validated": False}

