import asyncio
import subprocess
from fastmcp import Client

# Get the gateway service URL from minikube
result = subprocess.run(
    ["minikube", "service", "gateway", "-n", "apps", "--url"],
    capture_output=True,
    text=True,
    check=True
)
GATEWAY_URL = result.stdout.strip()


def get_mcp_client(sandbox_gateway, sandbox_uuid: str):
    config = {
        "mcpServers": {
            "sandbox": {
                "transport": "http",
                "url": f"{sandbox_gateway}/mcp",
                "headers": {"X-MCP-Session-ID": sandbox_uuid},
            }
        },
    }
    return Client(config, timeout=60)

mcp_client = get_mcp_client(
    GATEWAY_URL,
    "sandbox-i2kj3bzxdxgj-bd227356-912c-4c75-b080-d2e16584f7d8"
)

import json

def pretty_print_result(result):
    print("=== Call Tool Result ===")
    print(f"Is Error: {result.is_error}")

    if hasattr(result, 'content') and result.content:
        print("\n--- Content ---")
        for i, content in enumerate(result.content):
            print(f"Content {i+1}:")
            print(f"  Type: {content.type}")
            if hasattr(content, 'text') and content.text:
                print(f"  Text:\n{content.text}")

    if hasattr(result, 'structured_content') and result.structured_content:
        print("\n--- Structured Content ---")
        print(json.dumps(result.structured_content, indent=2))

    if hasattr(result, 'data') and result.data:
        print("\n--- Data ---")
        if hasattr(result.data, 'result'):
            print(f"Result: {result.data.result}")


async def main():
    async with mcp_client as client:
        tools = await client.list_tools()
        print("=== Available Tools ===")
        for i, tool in enumerate(tools, 1):
            print(f"\nTool {i}: {tool.name}")
            if tool.description:
                print(f"  Description: {tool.description.strip()}")

            print("  Input Schema:")
            if tool.inputSchema and 'properties' in tool.inputSchema:
                for prop_name, prop_info in tool.inputSchema['properties'].items():
                    required = prop_name in tool.inputSchema.get('required', [])
                    default = prop_info.get('default', 'N/A') if not required else 'N/A'
                    print(f"    - {prop_name} ({prop_info.get('type', 'unknown')}){' [required]' if required else ''}")
                    if default != 'N/A':
                        print(f"      Default: {default}")
                    if 'title' in prop_info:
                        print(f"      Title: {prop_info['title']}")

            if tool.outputSchema and 'properties' in tool.outputSchema:
                print("  Output Schema:")
                for prop_name, prop_info in tool.outputSchema['properties'].items():
                    print(f"    - {prop_name} ({prop_info.get('type', 'unknown')})")
        result = await client.call_tool("terminal-controller_execute_command", {"command": "ls"})
        pretty_print_result(result)


if __name__ == "__main__":
    asyncio.run(main())