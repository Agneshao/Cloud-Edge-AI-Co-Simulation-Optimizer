#!/usr/bin/env python3
"""Quick test to verify the MCP server tools work correctly."""

import asyncio
import sys
sys.path.insert(0, "mcp_servers/jetson_logs_server")

from server import handle_list_logs, handle_load_log

async def main():
    print("=== Testing list_logs ===")
    result = await handle_list_logs()
    print(result[0].text)

    print("\n=== Testing load_log ===")
    # Use the first available log file
    import json
    logs = json.loads(result[0].text)
    if logs["files"]:
        filename = logs["files"][0]["filename"]
        print(f"Loading: {filename}")
        result = await handle_load_log(filename)
        print(result[0].text)
    else:
        print("No log files found!")

if __name__ == "__main__":
    asyncio.run(main())
