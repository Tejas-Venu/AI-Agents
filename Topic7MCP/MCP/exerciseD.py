#!/usr/bin/env python3
"""
citation_explorer.py

Citation Network Explorer Agent.

See top-of-file docstring for description and usage.
"""

import os
import sys
import json
import re
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# === Configuration ===
MCP_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"
OPENAI_CHAT_COMPLETIONS = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

# === Helpers: MCP calls (robust to SSE 'data:' lines) ===


def _mcp_headers() -> Dict[str, str]:
    key = os.environ.get("ASTA_API_KEY", "")
    if not key:
        raise RuntimeError("ASTA_API_KEY environment variable not set")
    return {"Content-Type": "application/json", "x-api-key": key, "Accept": "application/json, text/event-stream"}


def _extract_data_payloads_from_sse(raw: str) -> List[str]:
    payloads = []
    if not raw:
        return payloads
    for line in raw.splitlines():
        if line.startswith("data:"):
            p = line[len("data:"):].strip()
            if p:
                payloads.append(p)
    return payloads


def _try_parse_json_safe(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_texts_from_mcp_response_obj(obj: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(obj, dict):
        res = obj.get("result")
        if isinstance(res, dict) and isinstance(res.get("content"), list):
            for c in res["content"]:
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str):
                        texts.append(t)
        if isinstance(res, str):
            texts.append(res)
        if isinstance(res, dict) and not texts:
            try:
                texts.append(json.dumps(res, ensure_ascii=False))
            except Exception:
                texts.append(str(res))
        if "content" in obj and isinstance(obj["content"], list):
            for c in obj["content"]:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    texts.append(c["text"])
        # If the object directly looks like a payload array or list, stringify it
        if "tools" in obj and isinstance(obj["tools"], list):
            texts.append(json.dumps(obj["tools"], ensure_ascii=False))
    elif isinstance(obj, str):
        texts.append(obj)
    return texts


def call_mcp_tool(name: str, arguments: Dict[str, Any], timeout: int = 60) -> str:
    """
    Call MCP tools/call with the given name and arguments.
    Returns a textual representation of the response (prefer content[*].text when present).
    Always returns a string (error strings are returned on failure).
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    try:
        resp = requests.post(MCP_ENDPOINT, headers=_mcp_headers(), json=payload, timeout=timeout)
    except Exception as e:
        return f"[ASTA ERROR] network error calling {name}: {e}"

    raw = resp.text or ""
    # Try parse JSON
    try:
        parsed = resp.json()
    except Exception:
        parsed = None

    if parsed is not None:
        # prefer textual content in common locations
        texts = _extract_texts_from_mcp_response_obj(parsed)
        if texts:
            return "\n\n".join(texts)
        # else return compact JSON string
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return str(parsed)

    # fallback: parse SSE-style data: payloads
    payloads = _extract_data_payloads_from_sse(raw)
    if payloads:
        collected_texts: List[str] = []
        for p in payloads:
            obj = _try_parse_json_safe(p)
            if obj is not None:
                collected_texts.extend(_extract_texts_from_mcp_response_obj(obj))
        if collected_texts:
            return "\n\n".join(collected_texts)
        return "\n".join(payloads)

    # final fallback: return raw body truncated
    return raw[:20000]


# === Small helpers to parse/normalize MCP responses ===


def parse_possible_json_blob(blob: str) -> Optional[Any]:
    """
    Many MCP textual fields themselves contain JSON strings (or arrays). Try to parse.
    """
    if not blob:
        return None
    blob = blob.strip()
    # If it looks like JSON, try parse
    if blob.startswith("{") or blob.startswith("["):
        return _try_parse_json_safe(blob)
    # sometimes blob contains multiple JSON objects separated by blank lines; try split
    pieces = re.split(r'\n\s*\n', blob)
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        parsed = _try_parse_json_safe(p)
        if parsed:
            return parsed
    return None


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


# === Retrieval pipeline functions ===


def get_seed_paper(paper_id: str) -> Dict[str, Any]:
    text = call_mcp_tool(
        "get_paper",
        {
            "paper_id": paper_id,
            "fields": "title,abstract,year,authors,fieldsOfStudy,references"
        }
    )
    parsed = parse_possible_json_blob(text)

    def extract_paper(obj):
        if isinstance(obj, dict):
            for key in ("paper", "result", "data", "response"):
                if key in obj:
                    found = extract_paper(obj[key])
                    if isinstance(found, dict):
                        return found
            if any(k in obj for k in ("title", "abstract", "authors", "year", "references")):
                return obj
            if "content" in obj and isinstance(obj["content"], list):
                for c in obj["content"]:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        p2 = parse_possible_json_blob(c["text"])
                        if isinstance(p2, dict):
                            return p2
        return None

    paper = extract_paper(parsed)
    if isinstance(paper, dict):
        return paper

    return {"_raw": text}


def get_paper_batch(ids: List[str], fields: str = "title,abstract,year,authors,citationCount") -> List[Dict[str, Any]]:
    """
    Use get_paper_batch if available to fetch metadata for multiple ids.
    If tool not available or fails, call get_paper per id.
    """
    # First try batch
    text = call_mcp_tool("get_paper_batch", {"ids": ids, "fields": fields})
    parsed = parse_possible_json_blob(text)
    results: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        # assume each element is a paper dict
        for p in parsed:
            if isinstance(p, dict):
                results.append(p)
        if results:
            return results
    # otherwise try to interpret as dict with 'papers' or 'results'
    if isinstance(parsed, dict):
        for key in ("papers", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                for p in parsed[key]:
                    if isinstance(p, dict):
                        results.append(p)
                if results:
                    return results
    # fallback: call get_paper individually
    for pid in ids:
        p = get_seed_paper(pid)
        if isinstance(p, dict):
            results.append(p)
    return results


def get_citations_for(paper_id: str, since_date: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Call get_citations for paper_id. Optional since_date in 'YYYY-MM-DD:' format to filter.
    Returns list of citing paper dicts (as best parsed).
    """
    args = {"paper_id": paper_id, "fields": "title,authors,publicationDate,year,abstract,doi", "limit": limit}
    if since_date:
        args["publication_date_range"] = since_date
    text = call_mcp_tool("get_citations", args)
    parsed = parse_possible_json_blob(text)
    items: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for it in parsed:
            if isinstance(it, dict):
                items.append(it)
    elif isinstance(parsed, dict):
        # look for common keys
        for key in ("citations", "papers", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                for it in parsed[key]:
                    if isinstance(it, dict):
                        items.append(it)
                if items:
                    break
        # also sometimes response is a list under result.content[*].text; handled by parse_possible_json_blob earlier
    else:
        # try to extract titles from text fallback
        # skip, return empty list
        pass
    return items


def get_author_papers(author_id: str, fields: str = "title,year,authors,citationCount,abstract,paperId", limit: int = 200) -> List[Dict[str, Any]]:
    """
    Call get_author_papers to get papers by an author.
    """
    text = call_mcp_tool("get_author_papers", {"author_id": author_id, "paper_fields": fields, "limit": limit})
    parsed = parse_possible_json_blob(text)
    out: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for it in parsed:
            if isinstance(it, dict):
                out.append(it)
    elif isinstance(parsed, dict):
        for key in ("papers", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                for it in parsed[key]:
                    if isinstance(it, dict):
                        out.append(it)
                if out:
                    break
        # sometimes the top-level parsed dict *is* the paper object
        if not out and "title" in parsed:
            out.append(parsed)
    return out


def search_papers_by_relevance(keyword: str, fields: str = "title,year,authors,citationCount,publicationDate,paperId", limit: int = 20) -> List[Dict[str, Any]]:
    text = call_mcp_tool("search_papers_by_relevance", {"keyword": keyword, "fields": fields, "limit": limit})
    parsed = parse_possible_json_blob(text)
    out: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for it in parsed:
            if isinstance(it, dict):
                out.append(it)
    elif isinstance(parsed, dict):
        for key in ("papers", "results", "data", "items"):
            if key in parsed and isinstance(parsed[key], list):
                for it in parsed[key]:
                    if isinstance(it, dict):
                        out.append(it)
                if out:
                    break
        # sometimes content is included in 'content' text as JSON
    return out


# === Utility selectors ===


def top_k_by_citation(papers: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    def citation_count(p):
        c = p.get("citationCount") or p.get("numCitations") or p.get("citation_count") or p.get("citation") or 0
        try:
            return int(c)
        except Exception:
            return 0
    sorted_p = sorted(papers, key=citation_count, reverse=True)
    return sorted_p[:k]


def parse_authors_list(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    # authors may be a list of dicts or list of names; normalize to list of dicts with at least name and id if present
    authors_raw = paper.get("authors") or []
    out = []
    if isinstance(authors_raw, list):
        for a in authors_raw:
            if isinstance(a, dict):
                out.append(a)
            elif isinstance(a, str):
                out.append({"name": a})
    return out


# === LLM: generate markdown report ===
def compress_for_llm(data):
    """
    Reduce the structured dataset so the LLM prompt stays small.
    Keeps only the fields needed for the markdown report.
    """

    seed = data.get("seed_paper", {})

    compact = {
        "seed_paper": {
            "title": seed.get("title"),
            "abstract": seed.get("abstract"),
            "year": seed.get("year"),
            "authors": [a.get("name") for a in seed.get("authors", [])]
        },

        "foundational_works": [
            {
                "title": p.get("title"),
                "year": p.get("year"),
                "abstract": (p.get("abstract") or "")[:500]
            }
            for p in data.get("foundational_works", [])[:5]
        ],

        "recent_developments": [
            {
                "title": p.get("title"),
                "year": p.get("year"),
                "abstract": (p.get("abstract") or "")[:500]
            }
            for p in data.get("recent_citing_papers", [])[:5]
        ],

        "author_profiles": [
            {
                "author": p.get("name"),
                "top_work": {
                    "title": p.get("top_other_work", {}).get("title"),
                    "year": p.get("top_other_work", {}).get("year"),
                    "citationCount": p.get("top_other_work", {}).get("citationCount"),
                }
            }
            for p in data.get("author_profiles", [])
        ],

        "recurring_authors": data.get("recurring_authors_overlap", [])
    }

    return compact

def call_openai_generate_report(structured_data: Dict[str, Any]) -> str:
    """
    Send collected structured_data to OpenAI (gpt-4o-mini) and ask for a markdown report.
    structured_data should be a JSON-serializable dict containing:
      seed_paper, foundational_works (list), recent_developments (list), author_profiles (list),
      overlapping_authors (list), notes (list)
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    system = {
        "role": "system",
        "content": "You are an academic research assistant that writes clear, structured Markdown reports for literature overviews."
    }

    # We will pass the structured data in a single user content blob. Keep it reasonably sized.
    user_content = (
        "Write a Markdown report using ONLY the JSON data below. "
        "Do not use placeholders like [Insert Title] or [Insert Year]. "
        "If a field is missing, write 'Unknown'. "
        "Summarize the actual titles, years, and abstracts that are present. "
        "Do not invent papers, authors, or citation counts."
        "\n\nStructured JSON:\n\n"
        + json.dumps(structured_data, ensure_ascii=False, indent=2)
    )

    messages = [system, {"role": "user", "content": user_content}]
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1400,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(OPENAI_CHAT_COMPLETIONS, headers=headers, json=payload, timeout=60)
    try:
        body = resp.json()
    except Exception:
        raise RuntimeError("OpenAI did not return JSON: " + (resp.text or "")[:2000])

    # Extract assistant content
    try:
        assistant_text = body["choices"][0]["message"]["content"]
        return assistant_text
    except Exception as e:
        # fallback: return whole response
        return json.dumps(body, ensure_ascii=False, indent=2)


# === Main agent workflow ===


def build_citation_neighborhood(seed_paper_id: str) -> Dict[str, Any]:
    """
    Main pipeline orchestrator. Returns a structured dictionary with all data needed for the LLM.
    """
    result: Dict[str, Any] = {"notes": []}

    # 1) Seed paper metadata
    seed = get_seed_paper(seed_paper_id)
    result["seed_paper_id"] = seed_paper_id
    result["seed_paper"] = seed

    if "_raw" in seed:
        result["notes"].append(f"Failed to parse seed paper metadata; raw response captured.")
        # still continue: seed may have minimal info only

    # 2) References: many MCP paper objects include a 'references' list of objects with 'paperId' or 'id'
    references = []
    refs_raw = seed.get("references") or []
    # references might be list of dicts containing paperId or citedPaper
    ref_ids = []
    for r in refs_raw:
        if isinstance(r, dict):
            for key in ("paperId", "paper_id", "id", "paperIdString", "citedPaper"):
                if key in r:
                    val = r[key]
                    if isinstance(val, str):
                        ref_ids.append(val)
                    elif isinstance(val, dict) and "paperId" in val:
                        ref_ids.append(val["paperId"])
                    break
            # sometimes the reference includes a nested 'title' but no id; skip those
        elif isinstance(r, str):
            ref_ids.append(r)
    # dedupe
    ref_ids = list(dict.fromkeys(ref_ids))
    result["reference_count_found"] = len(ref_ids)

    # fetch metadata for references (batch if possible)
    if ref_ids:
        ref_metas = get_paper_batch(ref_ids, fields="title,abstract,year,authors,citationCount,paperId")
    else:
        ref_metas = []
    result["all_references"] = ref_metas

    # pick top 5 foundational works by citation count
    top5_refs = top_k_by_citation(ref_metas, k=5)
    result["foundational_works"] = top5_refs

    # 3) Recent citing papers (last 3 years)
    three_years_ago = (datetime.utcnow() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    # MCP expects "YYYY-MM-DD:" to mean from that date to present
    since_range = f"{three_years_ago}:"
    citing = get_citations_for(seed_paper_id, since_date=since_range, limit=500)
    # sort citing by publicationDate/year descending, pick 5
    def pubdate_key(p):
        dt = p.get("publicationDate") or p.get("year") or ""
        if isinstance(dt, str):
            m = re.search(r'(19|20)\d{2}', dt)
            if m:
                return int(m.group(0))
        if isinstance(dt, int):
            return int(dt)
        return 0
    citing_sorted = sorted(citing, key=pubdate_key, reverse=True)
    recent5 = citing_sorted[:5]
    result["recent_citing_papers"] = recent5

    # 4) For each author of seed paper: retrieve their most-cited other work
    authors = parse_authors_list(seed)
    author_profiles = []
    for a in authors:
        # identify author id if present
        author_id = a.get("authorId") or a.get("authorIdString") or a.get("id") or a.get("userId") or a.get("author_id")
        author_name = a.get("name") or a.get("authorName") or a.get("displayName") or "Unknown"
        profile = {"name": author_name, "author_id": author_id}
        if not author_id:
            # attempt to search by name
            search_res = call_mcp_tool("search_authors_by_name", {"name": author_name, "fields": "authorId,name,affiliations", "limit": 5})
            parsed = parse_possible_json_blob(search_res)
            found_id = None
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict):
                    found_id = first.get("authorId") or first.get("id")
            if found_id:
                author_id = found_id
                profile["author_id"] = author_id

        if author_id:
            papers = get_author_papers(author_id, fields="title,year,citationCount,abstract,paperId", limit=200)
            # exclude the seed paper itself
            papers_nonseed = [p for p in papers if p.get("paperId") != seed_paper_id and p.get("paperId") != seed.get("paperId")]
            if not papers_nonseed and papers:
                papers_nonseed = papers
            top = top_k_by_citation(papers_nonseed, k=1)
            if top:
                top_p = top[0]
                profile["top_other_work"] = top_p
            else:
                profile["top_other_work"] = {}
        else:
            profile["top_other_work"] = {}
        author_profiles.append(profile)
    result["author_profiles"] = author_profiles

    # 5) Bonus: detect recurring collaborations (authors in references ∩ authors in recent citing papers)
    ref_author_names = set()
    for r in ref_metas:
        for au in parse_authors_list(r):
            n = (au.get("name") or "").strip()
            if n:
                ref_author_names.add(n.lower())
    citing_author_names = set()
    for c in recent5:
        for au in parse_authors_list(c):
            n = (au.get("name") or "").strip()
            if n:
                citing_author_names.add(n.lower())
    overlapping = sorted(list(ref_author_names.intersection(citing_author_names)))
    result["recurring_authors_overlap"] = overlapping

    return result


# === CLI & main ===


def pick_seed_from_topic(topic: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Search for papers by relevance on topic, then pick the highest-cited paper.
    Returns (paper_id, paper_metadata) or (None, None) on failure.
    """
    hits = search_papers_by_relevance(topic, fields="title,year,authors,citationCount,paperId", limit=30)
    if not hits:
        return None, None
    # prefer highest citationCount
    best = top_k_by_citation(hits, k=1)
    if not best:
        return None, None
    chosen = best[0]
    pid = chosen.get("paperId") or chosen.get("paper_id") or chosen.get("id") or chosen.get("paperIdString")
    return pid, chosen


def main():
    if len(sys.argv) < 2:
        print("Usage: python citation_explorer.py <ARXIV:... | topic:...>")
        sys.exit(1)

    arg = sys.argv[1].strip()
    seed_paper_id = None
    chosen_seed_meta = None

    if arg.lower().startswith("topic:"):
        topic = arg[len("topic:"):].strip()
        print(f"[INFO] Searching for seed paper on topic: {topic}")
        pid, meta = pick_seed_from_topic(topic)
        if not pid:
            print("[ERROR] Could not find a seed paper for the topic.")
            sys.exit(1)
        seed_paper_id = pid
        chosen_seed_meta = meta
        print(f"[INFO] Selected seed paper: {seed_paper_id} -- {meta.get('title') if meta else ''}")
    else:
        seed_paper_id = arg

    print(f"[INFO] Building citation neighborhood for seed: {seed_paper_id}")
    structured = build_citation_neighborhood(seed_paper_id)

    # attach chosen_seed_meta if available
    if chosen_seed_meta:
        structured["selected_by_topic_search"] = chosen_seed_meta

    # Send to LLM to create the markdown report
    print("[INFO] Sending collected data to the LLM to generate the Markdown report...")
    try:
        compact_data = compress_for_llm(structured)
        print(json.dumps(compact_data, indent=2, ensure_ascii=False))
        report_md = call_openai_generate_report(compact_data)
    except Exception as e:
        report_md = f"# Error generating report via LLM\n\n{str(e)}\n\nStructured JSON:\n\n{json.dumps(structured, indent=2, ensure_ascii=False)}"

    # print the report to stdout
    print("\n\n" + report_md + "\n\n")


if __name__ == "__main__":
    main()