import os
import json
import time
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import trafilatura

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Calls Tavily Search API and returns a list of results:
    [{ 'title': ..., 'url': ..., 'content': ..., 'score': ... }, ...]
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
    # Normalize to a list of results
    results = data.get("results", [])
    cleaned = []
    for r in results:
        cleaned.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": r.get("content") or "",  # Tavily’s snippet/summary
            "score": r.get("score") or 0,
        })
    return cleaned

def fetch_url_text(url: str, timeout: int = 25) -> str:
    """
    Fetches a URL and extracts readable text using trafilatura (preferred)
    with BeautifulSoup fallback.
    """
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and extracted.strip():
                return extracted.strip()
    except Exception:
        pass

    # Fallback: raw HTML -> strip tags
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:15000]  # guardrail
    except Exception:
        return ""

def rerank_results(results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank by Tavily score and length of extracted text.
    Returns top_k results.
    """
    ranked = sorted(
        results,
        key=lambda r: (r.get("score", 0), len(r.get("text", ""))),
        reverse=True
    )
    return ranked[:top_k]

def web_research(query: str, k: int = 5) -> List[Dict[str, str]]:
    results = tavily_search(query, max_results=k*2)  # fetch more initially
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
        time.sleep(0.7)

    # Apply reranking + keep top_k
    return rerank_results(enriched, top_k=k)
