import os
import fastmcp

main_mcp = fastmcp.FastMCP(name="MainAppLive")

remote_proxy = fastmcp.FastMCP.as_proxy(fastmcp.Client(f"http://127.0.0.1:3000/mcp"))
main_mcp.mount(remote_proxy)

if __name__ == "__main__":
    main_mcp.run(transport="streamable-http", host="0.0.0.0", port=3001)