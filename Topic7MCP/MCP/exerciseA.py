#!/usr/bin/env python3
"""
exerciseA_final.py

Sets Accept to accept both application/json and text/event-stream (server requirement),
then requests /tools/list and prints each tool name, one-line description,
and required/optional params.

Top answers to the exercise questions:
1) Use a search-style tool like `search_papers` to find papers about "transformer attention mechanisms".
2) Use an author/coauthor tool (e.g., `get_author_publications`, `get_coauthors`, or `search_authors`) to find who else published in the same area.
"""

import os
import sys
import json
import requests
from typing import Dict, Any, List, Optional

MCP_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"

def pretty_type(prop_schema: Dict[str, Any]) -> str:
    if not prop_schema:
        return "unknown"
    t = prop_schema.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1:
            return non_null[0] + " (nullable)"
        return "/".join(t)
    if isinstance(t, str):
        return t
    if "$ref" in prop_schema:
        return f"ref({prop_schema['$ref']})"
    if "enum" in prop_schema:
        return "enum"
    if "anyOf" in prop_schema:
        types = []
        for s in prop_schema.get("anyOf", []):
            if isinstance(s, dict) and "type" in s:
                types.append(str(s["type"]))
        return "anyOf[" + ",".join(types) + "]" if types else "anyOf"
    if "format" in prop_schema:
        return f"{prop_schema.get('format')}"
    return "object"

def extract_schema(tool_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = ["input_schema", "params_schema", "schema", "input", "inputSchema", "parameters", "params"]
    for key in candidates:
        if key in tool_obj and isinstance(tool_obj[key], dict):
            return tool_obj[key]
    if "tool" in tool_obj and isinstance(tool_obj["tool"], dict):
        for key in candidates:
            if key in tool_obj["tool"] and isinstance(tool_obj["tool"][key], dict):
                return tool_obj["tool"][key]
    return None

def parse_schema_params(schema: Dict[str, Any]):
    required_list = schema.get("required", []) if isinstance(schema.get("required", []), list) else []
    properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
    required_params = []
    optional_params = []
    for name, prop_schema in properties.items():
        p_type = pretty_type(prop_schema)
        entry = {"name": name, "type": p_type}
        if name in required_list:
            required_params.append(entry)
        else:
            optional_params.append(entry)
    if not properties and schema.get("type") == "array" and "items" in schema and isinstance(schema["items"], dict):
        items = schema["items"]
        props = items.get("properties", {})
        reqs = items.get("required", []) if isinstance(items.get("required", []), list) else []
        for name, prop_schema in props.items():
            p_type = pretty_type(prop_schema)
            entry = {"name": name, "type": p_type}
            if name in reqs:
                required_params.append(entry)
            else:
                optional_params.append(entry)
    return required_params, optional_params

def print_tool_info(tool: Dict[str, Any]) -> None:
    name = tool.get("name") or tool.get("id") or "<unnamed_tool>"
    description = tool.get("description") or tool.get("summary") or tool.get("doc") or ""
    print(f"Tool: {name}")
    if description:
        single_line_desc = " ".join(description.splitlines()).strip()
        print(f"  Description: {single_line_desc}")
    else:
        print("  Description: (no description provided)")

    schema = extract_schema(tool)
    if not schema:
        params_list = tool.get("parameters") or tool.get("args") or tool.get("inputs")
        if isinstance(params_list, list) and params_list:
            required = [p for p in params_list if p.get("required") is True]
            optional = [p for p in params_list if not p.get("required")]
            if required:
                print("  Required: " + ", ".join(f"{p.get('name')} ({p.get('type','unknown')})" for p in required))
            else:
                print("  Required: (none detected)")
            if optional:
                print("  Optional: " + ", ".join(f"{p.get('name')} ({p.get('type','unknown')})" for p in optional))
            else:
                print("  Optional: (none detected)")
        else:
            print("  Required: (unknown - no schema provided)")
            print("  Optional: (unknown - no schema provided)")
        print()
        return

    required_params, optional_params = parse_schema_params(schema)
    if required_params:
        print("  Required:")
        for p in required_params:
            print(f"    {p['name']} ({p['type']})")
    else:
        print("  Required: (none)")
    if optional_params:
        print("  Optional:")
        for p in optional_params:
            print(f"    {p['name']} ({p['type']})")
    else:
        print("  Optional: (none)")
    try:
        props = schema.get("properties", {})
        described = []
        for k, v in (props.items() if isinstance(props, dict) else []):
            desc = v.get("description")
            if isinstance(desc, str) and desc.strip():
                described.append((k, desc.strip()))
            if len(described) >= 3:
                break
        if described:
            print("  Notes (sample parameter descriptions):")
            for name, desc in described:
                one_line = " ".join(desc.splitlines())[:200]
                print(f"    {name}: {one_line}")
    except Exception:
        pass
    print()

def read_sse_and_collect_json(resp: requests.Response) -> Dict[str, Any]:
    collected = []
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                collected.append(payload)
            elif line.startswith("{") or line.startswith("["):
                collected.append(line)
        if not collected:
            return {}
        joined = "\n".join(collected)
        try:
            parsed = json.loads(joined)
            return parsed
        except Exception:
            try:
                parsed = json.loads(collected[-1])
                return parsed
            except Exception:
                return {"_raw_sse": joined}
    except Exception as e:
        return {"_sse_error": str(e)}

def main():
    api_key = os.environ.get("ASTA_API_KEY", "")
    if not api_key:
        print("ERROR: ASTA_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Content-Type": "application/json",
        # Accept both JSON and SSE per server requirement
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key
    }

    payload = {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}

    session = requests.Session()
    session.trust_env = False

    try:
        resp = session.post(MCP_ENDPOINT, headers=headers, json=payload, stream=True, timeout=20)
    except requests.exceptions.RequestException as e:
        print("ERROR: request failed:", repr(e), file=sys.stderr)
        sys.exit(2)

    ctype = resp.headers.get("Content-Type", "")
    try:
        if "text/event-stream" in ctype:
            parsed = read_sse_and_collect_json(resp)
        elif "application/json" in ctype or resp.text.lstrip().startswith(("{", "[")):
            parsed = resp.json()
        else:
            parsed = read_sse_and_collect_json(resp)
            if not parsed:
                try:
                    parsed = resp.json()
                except Exception:
                    parsed = {"_raw": resp.text}
    except Exception as e:
        print("ERROR: failed to parse response:", repr(e), file=sys.stderr)
        print("Response headers:", resp.headers, file=sys.stderr)
        print("Response text (truncated):", resp.text[:2000], file=sys.stderr)
        sys.exit(3)

    tools = None
    if isinstance(parsed, dict):
        tools = parsed.get("result", {}).get("tools") or parsed.get("tools") or parsed.get("result")
    if not tools and isinstance(parsed, dict):
        if "result" in parsed and isinstance(parsed["result"], dict) and "tools" in parsed["result"]:
            tools = parsed["result"]["tools"]
    if not tools:
        print("No 'tools' found in MCP response. Raw parsed response:")
        if isinstance(parsed, dict):
            print(json.dumps(parsed, indent=2))
        else:
            print(str(parsed))
        sys.exit(0)

    if not isinstance(tools, list):
        print("Unexpected tools structure; printing raw 'tools' entry:")
        print(json.dumps(tools, indent=2))
        sys.exit(0)

    for tool in tools:
        try:
            print_tool_info(tool)
        except Exception as e:
            print("Failed to parse tool entry:", e)
            print(json.dumps(tool, indent=2))

if __name__ == "__main__":
    main()