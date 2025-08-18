import json
from typing import Dict, Any, List
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from retrieval.websearch import web_research

def call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

def build_research_queries(task: str, plan: Dict[str, Any]) -> List[str]:
    """
    Generate 3–5 focused queries from the task and plan.
    You can make this smarter later (another small LLM call).
    """
    queries = [f"{task} best practices", f"{task} logistics checklist", f"{task} sponsorship ideas"]
    # Add per-workstream queries
    for ws in (plan.get("workstreams") or [])[:2]:
        name = ws.get("name", "")
        queries.append(f"{task} {name.lower()} tools and templates")
    # Deduplicate
    seen = set()
    uniq = []
    for q in queries:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq[:5]

def researcher_web_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses web search + fetch to gather real sources, then asks LLM to synthesize with citations.
    """
    llm: ChatGroq = state["_llm"]  # provided by main orchestrator
    task = state["task"]
    plan = state.get("plan", {})

    queries = build_research_queries(task, plan)
    sources: List[Dict[str, str]] = []
    for q in queries:
        sources.extend(web_research(q, k=3))

    # Keep top ~8 non-empty sources
    filtered = [s for s in sources if s.get("text")]
    filtered = filtered[:8]

    # Build a compact context for the LLM
    ctx_parts = []
    for i, s in enumerate(filtered, 1):
        ctx_parts.append(
            f"[{i}] TITLE: {s['title']}\nURL: {s['url']}\nEXTRACT:\n{s['text'][:1800]}\n"
        )
    context_block = "\n\n".join(ctx_parts)

    system = (
        "You are the Web Research Agent. Given a planning task and sources (with URLs), "
        "synthesize findings into structured JSON: resources/tools, estimates, validation checklists, "
        "and 5–7 open questions. Only use information supported by the provided sources. "
        "Return VALID JSON ONLY. Include citations by listing the numeric source IDs you used."
    )
    user = f"""
TASK: {task}

PLAN (JSON):
{json.dumps(plan, ensure_ascii=False, indent=2)}

SOURCES:
{context_block}

Return JSON with this exact shape:
{{
  "resources": [{{"workstream": "string", "tools": ["..."], "templates": ["..."], "citations": [1,2]}}],
  "estimates": [{{"workstream": "string", "effort": "S|M|L", "notes": "string", "citations": [3]}}],
  "validation_checklists": [{{"workstream": "string", "checklist": ["..."], "citations": [1,4]}}],
  "open_questions": ["...", "..."],
  "used_sources": [1,2,3]
}}
Only JSON. No extra commentary.
"""
    out = call_llm(llm, system, user)

    # Try to parse JSON; if it fails, provide a safe empty structure
    try:
        data = json.loads(out[out.find("{"):out.rfind("}")+1])
    except Exception:
        data = {
            "resources": [],
            "estimates": [],
            "validation_checklists": [],
            "open_questions": [],
            "used_sources": []
        }

    # Also return the raw sources for the Reporter to cite properly
    return {
        "web_sources": filtered,
        "research": data
    }
