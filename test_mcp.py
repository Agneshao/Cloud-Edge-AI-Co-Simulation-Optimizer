#!/usr/bin/env python3
"""Test EdgeTwin MCP tools with real data from SmartBucket."""

import json
import time
import urllib.request
import urllib.error

MCP_ENDPOINT = "https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp"
SESSION_ID = f"test-{int(time.time())}"


def mcp_request(method: str, params: dict = None, req_id: int = 1):
    """Send an MCP JSON-RPC request with SSE handling."""
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params:
        payload["params"] = params

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "mcp-session-id": SESSION_ID,
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(MCP_ENDPOINT, data=data, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            # Read SSE response
            content = response.read().decode('utf-8')

            # Parse SSE events - look for data: lines
            result = None
            for line in content.split('\n'):
                if line.startswith('data: '):
                    try:
                        result = json.loads(line[6:])
                    except json.JSONDecodeError:
                        pass

            return result if result else {"raw": content}
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8') if e.fp else ""
        return {"error": f"HTTP {e.code}: {e.reason}", "body": body}
    except Exception as e:
        return {"error": str(e)}


def extract_result(response):
    """Extract the text content from MCP response."""
    if "error" in response:
        return response
    if "result" in response:
        content = response["result"].get("content", [])
        if content and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except:
                return content[0]["text"]
    return response


def main():
    print(f"MCP Endpoint: {MCP_ENDPOINT}")
    print(f"Session ID: {SESSION_ID}")
    print()

    # Initialize
    print("=== Initializing MCP Session ===")
    init_result = mcp_request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-script", "version": "1.0.0"}
    })
    print(json.dumps(init_result, indent=2)[:500])
    print()

    # Small delay
    time.sleep(0.5)

    # Test list_logs
    print("=== Testing list_logs ===")
    result = mcp_request("tools/call", {
        "name": "list_logs",
        "arguments": {}
    }, req_id=2)
    print(json.dumps(extract_result(result), indent=2))
    print()

    # Test load_log
    print("=== Testing load_log ===")
    result = mcp_request("tools/call", {
        "name": "load_log",
        "arguments": {"filename": "inference_log_20251201_211136.csv"}
    }, req_id=3)
    print(json.dumps(extract_result(result), indent=2))
    print()

    # Test recommend_config
    print("=== Testing recommend_config ===")
    result = mcp_request("tools/call", {
        "name": "recommend_config",
        "arguments": {"filename": "inference_log_20251201_211136.csv"}
    }, req_id=4)
    print(json.dumps(extract_result(result), indent=2))
    print()

    # Test evaluate_config with AI
    print("=== Testing evaluate_config (with AI) ===")
    result = mcp_request("tools/call", {
        "name": "evaluate_config",
        "arguments": {
            "log_filename": "inference_log_20251201_211136.csv",
            "proposed_config": {
                "precision": "INT8",
                "batch_size": 2,
                "resolution": 416,
                "power_mode": "15W"
            }
        }
    }, req_id=5)
    print(json.dumps(extract_result(result), indent=2))
    print()

    print("Done!")


if __name__ == "__main__":
    main()
