import os
import time
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import trafilatura

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Call Tavily Search API and return normalized results:
    [{ 'title': str, 'url': str, 'content': str, 'score': float }]
    """
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set in environment.")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answers": False,
        "include_images": False,
        "include_raw_content": False,
        "include_domains": [],
        "exclude_domains": [],
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cleaned = []
    for r in data.get("results", []):
        cleaned.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": r.get("content") or "",   # Tavily snippet
            "score": r.get("score") or 0.0,
        })
    return cleaned

def fetch_url_text(url: str, timeout: int = 25) -> str:
    """
    Fetch and extract readable text using trafilatura; fallback to BeautifulSoup.
    """
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and extracted.strip():
                return extracted.strip()
    except Exception:
        pass

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:15000]
    except Exception:
        return ""

def _domain_score(url: str) -> float:
    """
    Lightweight domain heuristic: boost .org/.edu/.gov and known reputable domains.
    """
    url_l = url.lower()
    bonus = 0.0
    if any(t in url_l for t in [".gov", ".edu", ".ac.", ".org"]):
        bonus += 0.5
    if any(t in url_l for t in ["hbr.org", "nasa.gov", "who.int", "oecd.org", "un.org", "mit.edu", "stanford.edu"]):
        bonus += 0.3
    return bonus

def rerank_results(results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank by Tavily score + extracted length + domain heuristic.
    """
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("score", 0.0) + _domain_score(r.get("url", "")),
            len(r.get("text", "")),
        ),
        reverse=True,
    )
    return ranked[:top_k]

def web_research(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    High-level helper: search, then fetch each URL content; rerank; return top-k.
    Output: [{ 'title', 'url', 'snippet', 'text' }, ...]
    """
    results = tavily_search(query, max_results=max(5, k * 2))
    enriched = []
    for r in results:
        url = r["url"]
        text = fetch_url_text(url)
        enriched.append({
            "title": r["title"],
            "url": url,
            "snippet": r.get("content", ""),
            "text": text
        })
        # Be polite + avoid getting rate-limited
        time.sleep(0.5)
    return rerank_results(enriched, top_k=k)
