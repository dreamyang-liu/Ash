"""
Benchmark example - Measures sandbox spawn and MCP connection times.

Usage:
    python -m client.demo
"""
import asyncio
import logging
import time

from client import SandboxClient, SandboxConfig

logger = logging.getLogger(__name__)


async def demo():
    """Benchmark sandbox spawn and MCP connection times."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create config (reusable for multiple clients)
    config = SandboxConfig(
        control_plane_url="http://192.168.49.2:31786",
        gateway_url="http://192.168.49.2:31770",
        mcp_timeout=180,
    )

    # Use context manager for automatic cleanup
    with SandboxClient(config) as client:
        # Cleanup any existing sandboxes
        client.destroy_all()

        # Health check
        logger.info(f"Control plane healthy: {client.health_check()}")
        logger.info(f"Control plane ready: {client.ready_check()}")

        # Benchmark sandbox spawn
        logger.info("Starting sandbox spawn benchmark...")
        spawn_start = time.perf_counter()
        sandbox = client.spawn()
        spawn_elapsed = time.perf_counter() - spawn_start
        logger.info(f"Sandbox spawn time: {spawn_elapsed:.2f}s")
        logger.info(f"Created sandbox: {sandbox}")
        logger.info(f"  Host: {sandbox.host}")
        logger.info(f"  Ports: {sandbox.ports}")

        # Connect to it and benchmark MCP connection
        mcp = client.connect()

        logger.info("Starting MCP connection benchmark...")
        connect_start = time.perf_counter()
        async with mcp:
            connect_elapsed = time.perf_counter() - connect_start
            logger.info(f"MCP connection time: {connect_elapsed:.2f}s")

            await mcp.ping()
            logger.info("Connected to sandbox MCP")

            tools = await mcp.list_tools()
            logger.info(f"Available tools ({len(tools)}): {[t.name for t in tools[:5]]}...")

            result = await mcp.call_tool(
                "terminal-controller_execute_command",
                {"command": "sleep 60 && echo 'Hello from sandbox!'", "timeout": 61}
            )
            logger.info(f"Command result: {result.content[0].text if result.content else result}")

        # Summary
        total_time = spawn_elapsed + connect_elapsed
        logger.info("\n=== Benchmark Summary ===")
        logger.info(f"  Spawn time:      {spawn_elapsed:.2f}s")
        logger.info(f"  MCP connect:     {connect_elapsed:.2f}s")
        logger.info(f"  Total ready:     {total_time:.2f}s")

        # Sandbox automatically destroyed on context manager exit


if __name__ == "__main__":
    asyncio.run(demo())
