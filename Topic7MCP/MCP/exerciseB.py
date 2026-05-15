#!/usr/bin/env python3
"""
exerciseB_final.py

Robust implementation of Exercise B drills that handles SSE (Server-Sent Events),
JSON embedded inside `result["content"][*]["text"]`, noise filtering, and deduplication.

Drill 1 — search_papers_by_relevance:
  Query: "large language model agents"
  Print top 5 results as numbered list with title and year.

Drill 2 — get_citations:
  For ARXIV:1810.04805, filter from 2023 onward.
  Print count of citing papers parsed and titles of the first 5.

Drill 3 — get_paper (references):
  For ARXIV:2210.03629, request references and print their titles and years sorted by year ascending.

Usage:
  export ASTA_API_KEY="your_key_here"
  python exerciseB_final.py
"""
from typing import Any, Dict, List, Optional
import os
import re
import json
import sys
import requests

MCP_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"

# -------------------- SSE / MCP parsing helpers --------------------


def parse_sse_raw_to_events(raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract payloads from SSE-style responses. Collect JSON objects from lines
    beginning with 'data:'; return a list of parsed JSON objects or dicts with '_raw_data'.
    """
    events: List[Dict[str, Any]] = []
    if not raw_text:
        return events
    for line in raw_text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                events.append(json.loads(payload))
            except Exception:
                events.append({"_raw_data": payload})
    return events


def extract_text_contents_from_events(events: List[Dict[str, Any]]) -> List[Any]:
    """
    From parsed SSE events (JSON-RPC envelopes), extract each content[*]['text'] if present.
    Attempt to parse each text as JSON; otherwise include the raw text (may be multi-line).
    Returns a flattened list of parsed dicts/lists and raw strings.
    """
    outputs: List[Any] = []
    for ev in events:
        # ev often looks like {"jsonrpc":"2.0","id":...,"result":{"content":[{"type":"text","text":"..."}], ...}}
        rpc_res = ev.get("result") if isinstance(ev, dict) else None
        if isinstance(rpc_res, dict) and isinstance(rpc_res.get("content"), list):
            for c in rpc_res["content"]:
                if not isinstance(c, dict):
                    continue
                text = c.get("text")
                if not isinstance(text, str):
                    continue
                # Try parse text as JSON
                try:
                    parsed = json.loads(text)
                    outputs.append(parsed)
                    continue
                except Exception:
                    pass
                # If text contains multiple JSON objects or mixed content, split on blank lines and try pieces
                pieces = re.split(r'\n\s*\n', text.strip())
                parsed_any = False
                for piece in pieces:
                    piece = piece.strip()
                    if not piece:
                        continue
                    try:
                        parsed_piece = json.loads(piece)
                        outputs.append(parsed_piece)
                        parsed_any = True
                    except Exception:
                        # keep as raw piece for further filtering
                        outputs.append(piece)
                if parsed_any:
                    continue
                # fallback: raw text blob
                outputs.append(text)
        else:
            # No content list: include the event itself to let higher-level logic inspect it
            outputs.append(ev)
    return outputs


# -------------------- Filtering & normalization helpers --------------------


def filter_and_dedupe_title_lines(multi_line_text: str) -> List[str]:
    """
    Heuristic extraction of plausible title lines from a large multi-line text blob.
    Returns a list of cleaned, deduplicated title-like lines preserving order.
    """
    seen = set()
    keep: List[str] = []

    # Patterns that usually indicate noise we want to drop
    noise_patterns = re.compile(
        r'\b(Action|Observation|Result|Thought|Question|Search|Observation|pack of|B0061IVFZE|Milhouse|put knife|Coster-Waldau)\b',
        re.I,
    )

    # Lines that begin with a dash or em-dash/bullet often denote list entries
    candidate_re = re.compile(r'^\s*[-—•]\s*(.+)$')

    for raw_line in multi_line_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        m = candidate_re.match(raw_line)
        if m:
            line = m.group(1).strip()
        else:
            line = raw_line

        # Basic heuristics: require at least 3 words and some letters
        if len(line.split()) < 3:
            continue
        if not re.search(r'[A-Za-z]', line):
            continue
        if noise_patterns.search(line):
            continue

        # Normalize spacing and control chars
        cleaned = re.sub(r'[\x00-\x1f]+', ' ', line).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        keep.append(cleaned)

    return keep


def collect_papers_from_outputs(outputs: List[Any], title_keys=("title", "paperTitle", "name")) -> List[Dict[str, Any]]:
    """
    Given mixed outputs (dicts, lists, strings), produce a list of paper-like dicts that
    at least contain a 'title' key. Deduplicate by normalized title.
    """
    collected: List[Dict[str, Any]] = []
    seen_titles = set()

    for item in outputs:
        if isinstance(item, dict):
            # If the dict itself looks like a paper (has a title)
            title = None
            for k in title_keys:
                if k in item and item.get(k):
                    title = item.get(k)
                    break
            if title:
                norm = str(title).strip().lower()
                if norm not in seen_titles:
                    seen_titles.add(norm)
                    # Keep the raw dict but ensure 'title' exists
                    entry = dict(item)
                    entry["title"] = title
                    collected.append(entry)
                continue

            # If it contains nested lists, attempt to extract papers from those lists
            for v in item.values():
                if isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, dict):
                            stitle = None
                            for k in title_keys:
                                if k in sub and sub.get(k):
                                    stitle = sub.get(k)
                                    break
                            if stitle:
                                norm = str(stitle).strip().lower()
                                if norm not in seen_titles:
                                    seen_titles.add(norm)
                                    collected.append(sub)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    stitle = None
                    for k in title_keys:
                        if k in sub and sub.get(k):
                            stitle = sub.get(k)
                            break
                    if stitle:
                        norm = str(stitle).strip().lower()
                        if norm not in seen_titles:
                            seen_titles.add(norm)
                            collected.append(sub)
        elif isinstance(item, str):
            # It's a text blob; extract plausible title lines
            title_lines = filter_and_dedupe_title_lines(item)
            for t in title_lines:
                norm = t.lower()
                if norm not in seen_titles:
                    seen_titles.add(norm)
                    collected.append({"title": t})

    return collected


# -------------------- MCP call helper --------------------


def call_tool_raw(session: requests.Session, name: str, arguments: Dict[str, Any], rpc_id: int = 1, timeout: int = 30) -> List[Any]:
    """
    Call MCP 'tools/call' and return parsed inner outputs:
      - Parse SSE 'data:' lines into JSON objects
      - Extract content[*]['text'] strings and parse them as JSON where possible
      - Return a flattened list of parsed dicts/lists and raw strings
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": os.environ.get("ASTA_API_KEY", ""),
    }
    if not headers["x-api-key"]:
        raise RuntimeError("ASTA_API_KEY environment variable not set")

    payload = {"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    resp = session.post(MCP_ENDPOINT, headers=headers, json=payload, timeout=timeout)

    raw = resp.text or ""
    events = parse_sse_raw_to_events(raw)
    if not events:
        # fallback: try parse full body as JSON
        try:
            obj = resp.json()
            events = [obj]
        except Exception:
            # return raw body as single string if nothing parseable
            return [raw]

    outputs = extract_text_contents_from_events(events)
    return outputs


# -------------------- Drills --------------------


def drill_1_search_papers(session: requests.Session):
    """Drill 1 — search_papers: Find recent LLM agent papers (top 5)."""
    print("\nDrill 1 — search_papers: top 5 for 'large language model agents'\n")
    args = {
        # tool available on this MCP: search_papers_by_relevance expects 'keyword'
        "keyword": "large language model agents",
        "fields": "title,abstract,year,authors",
        "limit": 10,
    }
    outputs = call_tool_raw(session, "search_papers_by_relevance", args, rpc_id=2, timeout=30)
    papers = collect_papers_from_outputs(outputs)
    if not papers:
        print("No papers found. Raw outputs:")
        print(json.dumps(outputs, indent=2) if not isinstance(outputs, str) else outputs)
        return
    # Print top 5 with title and year (year may be absent)
    for i, p in enumerate(papers[:5], start=1):
        title = p.get("title", "<no title>")
        year = p.get("year") or p.get("publicationDate") or ""
        if isinstance(year, str) and len(year) >= 4:
            year = year[:4]
        print(f"{i}. {title} ({year})")


def drill_2_get_citations(session: requests.Session):
    """Drill 2 — get_citations: Trace impact of BERT (ARXIV:1810.04805) from 2023 onward."""
    print("\nDrill 2 — get_citations for ARXIV:1810.04805 (2023 onward)\n")
    args = {
        "paper_id": "ARXIV:1810.04805",
        "fields": "title,year,authors",
        "limit": 200,
        "publication_date_range": "2023-01-01:",
    }
    outputs = call_tool_raw(session, "get_citations", args, rpc_id=3, timeout=60)

    # Gather citing papers from mixed outputs
    citing = []
    for out in outputs:
        if isinstance(out, dict):
            # Common single-paper shape seen earlier: {"citingPaper": {...}}
            if "citingPaper" in out and isinstance(out["citingPaper"], dict):
                citing.append(out["citingPaper"])
                continue
            # Common list shapes
            for key in ("citations", "papers", "results", "data"):
                if key in out and isinstance(out[key], list):
                    citing.extend(out[key])
                    break
            # If out itself looks like a paper
            if "title" in out and ("year" in out or "authors" in out):
                citing.append(out)
        elif isinstance(out, list):
            for sub in out:
                if isinstance(sub, dict):
                    if "title" in sub:
                        citing.append(sub)
        elif isinstance(out, str):
            # maybe raw text with titles; try to extract title lines
            titles = filter_and_dedupe_title_lines(out)
            for t in titles:
                citing.append({"title": t})

    # Deduplicate by normalized title
    deduped = []
    seen = set()
    for c in citing:
        title = (c.get("title") or "").strip() if isinstance(c, dict) else str(c)
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    print(f"Total citing items parsed: {len(deduped)}")
    for i, p in enumerate(deduped[:5], start=1):
        title = p.get("title") if isinstance(p, dict) else str(p)
        print(f"{i}. {title}")


def drill_3_get_references(session: requests.Session):
    """
    Drill 3 — get_references:
    Understand the intellectual foundation of the ReAct paper (ARXIV:2210.03629).

    Fetch references, clean noisy results, deduplicate titles,
    sort by year ascending, and print "year — title".
    """

    print("\nDrill 3 — get_references for ARXIV:2210.03629 (sorted by year ascending)\n")

    args = {
        "paper_id": "ARXIV:2210.03629",
        "fields": "title,year,authors",
        "limit": 500
    }

    outputs = call_tool_raw(session, "get_references", args, rpc_id=4, timeout=60)

    refs = []

    # ---------------- Extract references ----------------
    for out in outputs:

        if isinstance(out, dict):

            if "references" in out and isinstance(out["references"], list):
                refs.extend(out["references"])

            elif "data" in out and isinstance(out["data"], list):
                refs.extend(out["data"])

            elif "results" in out and isinstance(out["results"], list):
                refs.extend(out["results"])

            elif "title" in out:
                refs.append(out)

        elif isinstance(out, list):
            for item in out:
                if isinstance(item, dict) and "title" in item:
                    refs.append(item)

    # ---------------- Clean + dedupe ----------------
    cleaned_refs = []
    seen_titles = set()

    for r in refs:

        if not isinstance(r, dict):
            continue

        title = (r.get("title") or "").strip()
        year = r.get("year")

        if not title:
            continue

        # Remove noisy lines accidentally parsed as references
        if re.search(r'\b(Action|Observation|Thought|Result|Search|Milhouse|B0061IVFZE)\b', title, re.I):
            continue

        norm = title.lower()

        if norm in seen_titles:
            continue

        seen_titles.add(norm)

        cleaned_refs.append({
            "title": title,
            "year": year
        })

    # ---------------- Sort by year ----------------
    def year_key(r):
        y = r.get("year")

        if isinstance(y, int):
            return y

        if isinstance(y, str):
            m = re.search(r'(19|20)\d{2}', y)
            if m:
                return int(m.group())

        return 9999

    refs_sorted = sorted(cleaned_refs, key=year_key)

    # ---------------- Print results ----------------
    for r in refs_sorted:
        year = r.get("year") or ""
        title = r.get("title")
        print(f"{year} — {title}")


# -------------------- Main --------------------


def main():
    session = requests.Session()
    # Avoid accidental proxying on clusters
    session.trust_env = False

    try:
        drill_1_search_papers(session)
    except Exception as e:
        print("Drill 1 failed:", e, file=sys.stderr)

    try:
        drill_2_get_citations(session)
    except Exception as e:
        print("Drill 2 failed:", e, file=sys.stderr)

    try:
        drill_3_get_references(session)
    except Exception as e:
        print("Drill 3 failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()