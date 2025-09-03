import os
from fastmcp import FastMCP

def get_config():
    return {
        "mcpServers": {
            "fetch": {
                "command": "python",
                "args": ["-m", "mcp_server_fetch", "--ignore-robots-txt", "--user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'"]
            }
        }
    }

def main():
    config = get_config()
    proxy = FastMCP.as_proxy(config, name="General MCP Proxy")
    proxy.run(transport="streamable-http", host="0.0.0.0", port=3000)

if __name__ == "__main__":
    main()