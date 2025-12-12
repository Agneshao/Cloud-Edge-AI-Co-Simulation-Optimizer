#!/bin/bash
# Test EdgeTwin MCP tools - run from WSL

MCP_URL="https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp"

echo "=== 1. Initialize MCP Session ==="
# Get session ID from response header
INIT_RESPONSE=$(curl -s -i -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}')

# Extract session ID from headers
SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i "mcp-session-id:" | tr -d '\r' | cut -d' ' -f2)

if [ -z "$SESSION_ID" ]; then
  echo "Failed to get session ID"
  echo "$INIT_RESPONSE"
  exit 1
fi

echo "Session ID: $SESSION_ID"
echo ""

echo "=== 2. List Logs ==="
curl -s -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_logs","arguments":{}}}' \
  | grep -o 'data: {.*}' | tail -1 | sed 's/data: //' | jq .
echo ""

echo "=== 3. Load Log ==="
curl -s -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"load_log","arguments":{"filename":"inference_log_20251201_211136.csv"}}}' \
  | grep -o 'data: {.*}' | tail -1 | sed 's/data: //' | jq '.result.content[0].text | fromjson'
echo ""

echo "=== 4. Recommend Config ==="
curl -s -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"recommend_config","arguments":{"filename":"inference_log_20251201_211136.csv"}}}' \
  | grep -o 'data: {.*}' | tail -1 | sed 's/data: //' | jq '.result.content[0].text | fromjson'
echo ""

echo "=== Done ==="
