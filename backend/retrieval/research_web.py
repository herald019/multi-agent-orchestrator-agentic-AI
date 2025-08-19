import json
from typing import Dict, Any, List
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from retrieval.websearch import web_research

def _call_llm(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return msg.content.strip()

def _build_research_queries(task: str, plan: Dict[str, Any]) -> List[str]:
    queries = [f"{task} best practices", f"{task} logistics checklist", f"{task} risk management"]
    for ws in (plan.get("workstreams") or [])[:2]:
        name = ws.get("name", "")
        if name:
            queries.append(f"{task} {name.lower()} tools and templates")
    # Dedup + limit
    uniq, seen = [], set()
    for q in queries:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq[:5]

def _summarize_source(llm: ChatGroq, source: Dict[str, str]) -> str:
    system = (
        "Summarize the source text in 4–6 factual sentences. "
        "Do NOT add info not present in the text."
    )
    user = f"TITLE: {source['title']}\nURL: {source['url']}\nTEXT:\n{source['text'][:4000]}"
    try:
        summary = _call_llm(llm, system, user)
        return summary.strip()
    except Exception:
        return (source.get('snippet') or source.get('text', ''))[:500]

def researcher_web_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm: ChatGroq = state["_llm"]
    task = state["task"]
    plan = state.get("plan", {})

    queries = _build_research_queries(task, plan)
    sources: List[Dict[str, str]] = []
    for q in queries:
        sources.extend(web_research(q, k=3))

    filtered = [s for s in sources if s.get("text")]
    filtered = filtered[:8]

    summaries = []
    for i, s in enumerate(filtered, 1):
        summary = _summarize_source(llm, s)
        summaries.append(
            f"[{i}] TITLE: {s['title']}\nURL: {s['url']}\nSUMMARY:\n{summary}\n"
        )
    context_block = "\n\n".join(summaries)

    system = (
        "You are the Web Research Agent. Using the provided sources, synthesize findings into structured JSON: "
        "resources/tools, estimates, validation checklists, and 5–7 open questions. "
        "Only include facts grounded in the sources. "
        "Return VALID JSON ONLY and include citations by listing numeric source IDs."
    )
    user = f"""
TASK: {task}

PLAN (JSON):
{json.dumps(plan, ensure_ascii=False, indent=2)}

SOURCES:
{context_block}

Return JSON exactly with this shape:
{{
  "resources": [{{"workstream": "string", "tools": ["..."], "templates": ["..."], "citations": [1,2]}}],
  "estimates": [{{"workstream": "string", "effort": "S|M|L", "notes": "string", "citations": [3]}}],
  "validation_checklists": [{{"workstream": "string", "checklist": ["..."], "citations": [1,4]}}],
  "open_questions": ["...", "..."],
  "used_sources": [1,2,3]
}}
Only JSON. No extra commentary.
"""
    out = _call_llm(llm, system, user)

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

    return {
        "web_sources": filtered,
        "research": data
    }
