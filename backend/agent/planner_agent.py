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

def _try_parse_json(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def _safe_call_planner(llm, system, user, retries=2):
    out = _call_llm(llm, system, user)
    data = _try_parse_json(out)
    if data and all(k in data for k in ["timeline", "workstreams", "risks", "metrics", "assumptions"]):
        return data

    for _ in range(retries):
        tightened = user + (
            "\n\nREMINDER: You MUST include >=4 timeline entries, >=4 risks, >=3 metrics, "
            "and >=4 workstreams with tasks/owner/dependencies. Output VALID JSON ONLY."
        )
        out = _call_llm(llm, system, tightened)
        data = _try_parse_json(out)
        if data and all(k in data for k in ["timeline", "workstreams", "risks", "metrics", "assumptions"]):
            return data
    return None

def planner_node(state: OrchestratorState) -> OrchestratorState:
    llm: ChatGroq = state["_llm"]
    _append_log(state, "Planner: creating detailed timeline, workstreams, and risks.")

    system = (
        "You are the Planner Agent. Produce comprehensive, realistic project plans.\n"
        "Constraints:\n"
        "- 4–6 timeline phases or weekly buckets, each with 2–4 milestones and deliverables.\n"
        "- 4–6 workstreams: e.g., Discovery/Research, Execution/Build, QA/Validation, Logistics/Operations, "
        "Comms/Marketing, Governance/Risk.\n"
        "- Each workstream: multiple tasks, an owner role, and explicit dependencies.\n"
        "- >=3 assumptions; >=4 risks (with impact + mitigation); >=3 success metrics.\n"
        "Output VALID JSON ONLY."
    )

    user = f"""
TASK: {state['task']}

Return JSON exactly in this shape:

{{
  "objective": "string",
  "assumptions": ["string", "string", "..."],
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
  "metrics": ["string", "string", "string"]
}}
"""
    data = _safe_call_planner(llm, system, user)
    if not data:
        data = {
            "objective": state["task"],
            "assumptions": ["TBD", "TBD", "TBD"],
            "timeline": [],
            "workstreams": [],
            "risks": [],
            "metrics": []
        }
    _append_log(state, "Planner: detailed plan drafted.")
    return {"plan": data}
