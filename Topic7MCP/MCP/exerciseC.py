#!/usr/bin/env python3
"""
asta_chatbot_full_fixed.py

Exercise C — Asta-Powered Research Chatbot (full script, fixed).

Fixes included:
- Robust handling of MCP tools/list (SSE streaming responses).
- Mapping sanitized function names back to original MCP tool names.
- Ensures every appended message to OpenAI has a string 'content' (no None).
- Robust parsing of function-call arguments and MCP responses.
- Prints which tool is being called and with what args.
- Graceful error handling: MCP call errors returned as tool results.

Requirements:
  pip install requests

Environment variables:
  ASTA_API_KEY   - API key for Asta MCP
  OPENAI_API_KEY - API key for OpenAI

Run:
  python asta_chatbot_full_fixed.py
"""

import os
import json
import re
import requests
from typing import Any, Dict, List, Optional, Tuple

# Config
MCP_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"
OPENAI_CHAT_COMPLETIONS = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOOL_CALL_LOOPS = 6

SYSTEM_PROMPT = (
    "You are a research assistant with programmatic access to Semantic Scholar-like Asta tools. "
    "You can call tools to find papers, get citations, fetch references, and more. Use function "
    "calls when appropriate. When you call a tool, include the exact arguments needed. When you "
    "have sufficient information, provide a concise, helpful answer."
)


# -------------------- Helpers for MCP & SSE parsing --------------------

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
        if "tools" in obj and isinstance(obj["tools"], list):
            try:
                texts.append(json.dumps(obj["tools"], ensure_ascii=False))
            except Exception:
                texts.append(str(obj["tools"]))
        if "content" in obj and isinstance(obj["content"], list):
            for c in obj["content"]:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    texts.append(c["text"])
    elif isinstance(obj, str):
        texts.append(obj)
    return texts


# -------------------- Robust MCP tools/list (SSE-friendly) --------------------

def mcp_tools_list() -> List[Dict[str, Any]]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = requests.post(MCP_ENDPOINT, headers=_mcp_headers(), json=payload, timeout=30)

    raw = resp.text or ""
    # Fast path: try resp.json()
    try:
        parsed = resp.json()
    except Exception:
        parsed = None

    def extract_tools_from_obj(obj) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("result"), list):
            return obj["result"]
        if isinstance(obj.get("tools"), list):
            return obj["tools"]
        res = obj.get("result")
        if isinstance(res, dict):
            if isinstance(res.get("tools"), list):
                return res["tools"]
            if isinstance(res.get("content"), list):
                for c in res["content"]:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        txt = c["text"].strip()
                        p = _try_parse_json_safe(txt)
                        if isinstance(p, list):
                            return p
                        if isinstance(p, dict) and isinstance(p.get("tools"), list):
                            return p["tools"]
        return None

    if parsed is not None:
        tools = extract_tools_from_obj(parsed)
        if tools is not None:
            return tools

    # SSE path
    data_payloads = _extract_data_payloads_from_sse(raw)
    for p in data_payloads:
        obj = _try_parse_json_safe(p)
        if obj is None:
            continue
        tools = extract_tools_from_obj(obj)
        if tools is not None:
            return tools

    if data_payloads:
        joined = "\n".join(data_payloads)
        obj = _try_parse_json_safe(joined)
        if obj is not None:
            tools = extract_tools_from_obj(obj)
            if tools is not None:
                return tools

    # Deep heuristic search in parsed
    if parsed is not None:
        def find_tool_list(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
                        if all(("name" in it or "tool" in it) for it in v):
                            return v
                    found = find_tool_list(v)
                    if found:
                        return found
            elif isinstance(o, list):
                for item in o:
                    found = find_tool_list(item)
                    if found:
                        return found
            return None

        found = find_tool_list(parsed)
        if found:
            return found

    preview = raw[:4000].replace("\n", "\\n")
    raise RuntimeError("Unexpected MCP tools/list response shape (raw preview):\n" + preview)


# -------------------- Convert MCP schema -> OpenAI function spec & mapping --------------------

def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    if not re.match(r'^[A-Za-z]', sanitized):
        sanitized = "f_" + sanitized
    return sanitized.lower()


def mcp_tool_to_openai_func_and_map(mcp_tool: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    orig_name = mcp_tool.get("name") or mcp_tool.get("tool") or "unknown_tool"
    description = mcp_tool.get("description", "") or ""
    parameters = mcp_tool.get("inputSchema")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    sanitized = _sanitize_name(orig_name)
    func = {"name": sanitized, "description": description, "parameters": parameters}
    return func, orig_name


def get_asta_tools() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    raw_tools = mcp_tools_list()
    functions: List[Dict[str, Any]] = []
    name_map: Dict[str, str] = {}
    for t in raw_tools:
        try:
            func, orig = mcp_tool_to_openai_func_and_map(t)
            functions.append(func)
            name_map[func["name"]] = orig
        except Exception as e:
            print(f"[WARN] skipping tool due to conversion error: {e}")
    print(f"[INFO] Loaded {len(functions)} functions from MCP")
    return functions, name_map


# -------------------- Call MCP tool (robust) --------------------

def call_asta_tool(mcp_tool_name: str, arguments: Dict[str, Any]) -> str:
    print(f"[ASTA] Calling tool: {mcp_tool_name} with args: {json.dumps(arguments, ensure_ascii=False)}")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": mcp_tool_name, "arguments": arguments}}
    try:
        resp = requests.post(MCP_ENDPOINT, headers=_mcp_headers(), json=payload, timeout=60)
    except Exception as e:
        err = f"[ASTA ERROR] network error calling tool {mcp_tool_name}: {e}"
        print(err)
        return err

    raw = resp.text or ""
    try:
        parsed = resp.json()
    except Exception:
        parsed = None

    if parsed is not None:
        texts = _extract_texts_from_mcp_response_obj(parsed)
        if texts:
            return "\n\n".join(texts)
        if isinstance(parsed, (dict, list)):
            try:
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return str(parsed)

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

    return raw[:20000]


# -------------------- OpenAI chat wrapper --------------------

def call_openai_chat(messages: List[Dict[str, Any]], functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "functions": functions,
        "function_call": "auto",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(OPENAI_CHAT_COMPLETIONS, headers=headers, json=payload, timeout=60)
    try:
        return resp.json()
    except Exception:
        raise RuntimeError("OpenAI returned non-JSON: " + (resp.text or "")[:2000])


# -------------------- Chat loop (single turn) --------------------

def chat(user_message: str, functions: List[Dict[str, Any]], name_map: Dict[str, str]) -> Optional[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for iter_i in range(MAX_TOOL_CALL_LOOPS):
        resp = call_openai_chat(messages, functions)
        if not isinstance(resp, dict) or "choices" not in resp or not resp["choices"]:
            print("[ERROR] Invalid OpenAI response:", resp)
            return None

        choice = resp["choices"][0]
        message = choice.get("message", {}) or {}

        if message.get("function_call"):
            fn_call = message["function_call"]
            fn_name = fn_call.get("name")
            fn_args_raw = fn_call.get("arguments") or "{}"

            try:
                if isinstance(fn_args_raw, str):
                    fn_args = json.loads(fn_args_raw)
                else:
                    fn_args = fn_args_raw
            except Exception:
                fn_args = {"_raw": fn_args_raw}

            print(f"[MODEL] Requested tool: {fn_name} with args: {json.dumps(fn_args, ensure_ascii=False)}")

            mcp_name = name_map.get(fn_name, fn_name)

            try:
                tool_result_text = call_asta_tool(mcp_name, fn_args)
            except Exception as e:
                tool_result_text = f"[ASTA ERROR] calling tool {mcp_name} failed: {e}"
                print(tool_result_text)

            if tool_result_text is None:
                tool_result_text = ""

            # IMPORTANT: assistant message content must be a string (not None)
            assistant_call_note = f"[Called tool {fn_name} -> MCP: {mcp_name} with args {json.dumps(fn_args, ensure_ascii=False)}]"
            messages.append({"role": "assistant", "content": assistant_call_note})

            # function message must have string content
            messages.append({"role": "function", "name": fn_name, "content": str(tool_result_text)})

            continue

        assistant_text = message.get("content", "")
        print("\n[ASSISTANT ANSWER]")
        print(assistant_text)
        return assistant_text

    print("[WARN] Reached maximum tool-call iterations without final answer.")
    return None


# -------------------- Main: startup and interactive CLI --------------------

def main():
    print("Starting Asta-Powered Research Chatbot (fixed)...")
    try:
        functions, name_map = get_asta_tools()
    except Exception as e:
        print("[FATAL] Failed to load tools from MCP:", e)
        return

    print(f"[INFO] {len(functions)} function specs available. Sample:")
    for f in functions[:10]:
        print(" -", f["name"], "|", (f.get("description") or "")[:120])

    tests = [
        "Find recent papers about large language model agents",
        "Who wrote Attention is All You Need and what else have they published?",
        "What papers cite the original BERT paper?",
        "Summarize the references used in the ReAct paper",
    ]
    print("\nTest queries:")
    for i, t in enumerate(tests, 1):
        print(f"{i}. {t}")

    while True:
        try:
            q = input("\nEnter a query (blank to run first test, 'quit' to exit): ").strip()
        except EOFError:
            break
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            q = tests[0]
            print("[AUTO] Running:", q)
        try:
            chat(q, functions, name_map)
        except Exception as e:
            print("[ERROR] Chat failed:", e)


if __name__ == "__main__":
    main()