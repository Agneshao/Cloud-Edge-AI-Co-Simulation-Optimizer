#!/bin/bash
# Upload local CSV logs to EdgeTwin SmartBucket via MCP

MCP_ENDPOINT="https://mcp-01kc4gy6ky0zj3cwk08ygbf913.01kbx9v4pvd1ecabcqmtnjt8q0.lmapp.run/mcp"
SESSION_ID="upload-session-$(date +%s)"

# Initialize session
echo "Initializing MCP session..."
curl -s -X POST "$MCP_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"upload-script","version":"1.0.0"}}}' > /dev/null

# Upload each CSV file
for csvfile in data/sample_logs/*.csv; do
    if [ -f "$csvfile" ]; then
        filename=$(basename "$csvfile")
        echo "Uploading: $filename"

        # Read CSV content and escape for JSON
        csv_content=$(cat "$csvfile" | jq -Rs .)

        # Call upload_log tool
        result=$(curl -s -X POST "$MCP_ENDPOINT" \
          -H "Content-Type: application/json" \
          -H "mcp-session-id: $SESSION_ID" \
          -d "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"upload_log\",\"arguments\":{\"filename\":\"$filename\",\"csv_content\":$csv_content}}}")

        # Check for success
        if echo "$result" | grep -q '"success": true'; then
            echo "  ✓ Uploaded successfully"
        else
            echo "  ✗ Upload failed"
            echo "$result" | jq .
        fi
    fi
done

echo ""
echo "Verifying uploads..."
curl -s -X POST "$MCP_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_logs","arguments":{}}}' | jq '.result.content[0].text | fromjson'

echo ""
echo "Done! Logs are now stored in SmartBucket."
