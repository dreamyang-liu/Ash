import os
from fastmcp import FastMCP

def get_config():
    mcp_hub_addr = os.getenv('MCP_HUB_ADDR')
    return {
        "mcpServers": {
            "remote": {
                "url": f"http://{mcp_hub_addr}",
                "transport": "http"
            },
            "terminal-controller": {
                "command": "python",
                "args": ["-m", "terminal_controller"]
            }
        }
    }

def main():
    config = get_config()
    proxy = FastMCP.as_proxy(config, name="General MCP Proxy")
    proxy.run(transport="streamable-http", host="0.0.0.0", port=3000)

if __name__ == "__main__":
    main()