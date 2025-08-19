import os
import fastmcp

main_mcp = fastmcp.FastMCP(name="MainAppLive")

mcp_addrs = os.getenv("MCP_ADDRS")
if mcp_addrs:
    mcp_list = mcp_addrs.split(",")
else:
    mcp_list = []

for mcp in mcp_list:
    remote_proxy = fastmcp.FastMCP.as_proxy(fastmcp.Client(f"http://{mcp}"))
    main_mcp.mount(remote_proxy)

if __name__ == "__main__":
    main_mcp.run(transport="streamable-http", host="0.0.0.0", port=3000)