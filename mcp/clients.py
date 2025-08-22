import abc
import asyncio
import json
import sys
import logging
from abc import ABC, abstractmethod

from asyncio.subprocess import Process
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, TextIO, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters
from mcp import Tool as MCPTool
from mcp import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult


logger = logging.getLogger(__name__)

class MCPClient(ABC):
    """Abstract base class for a MCP Client.

    This class defines the interface for a MCP client, without imposing implementation
    approach (e.g., Anthropic's ClientSession), or transport (e.g., stdio, SSE, etc.).
    """

    @abstractmethod
    async def connect(self):
        """Connect to the server.

        This can spawn a subprocess or open a network connection, depending
        on the transport used in the subclass's implementation.

        The server will remain connected until `cleanup()` is called.
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup the server connection.

        This can close a subprocess or close a network connection, depending
        on the transport used in the subclass's implementation.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """A readable name for the connection to the server.

        This is used for logging and debugging purposes.
        """
        pass

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the connected server.

        Returns:
            A list of `MCPTool` objects.
        """
        pass

    @abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult:
        """Invoke a tool on the connected server.

        Args:
            tool_name: The name of the tool to invoke.
            arguments: The arguments to pass to the tool.

        Returns:
            The result of the tool call.
        """
        pass


class MCPClientBase(MCPClient, abc.ABC):
    """Base MCP Client that uses `ClientSession` to communicate with the server.

    This class provides the core implementation of a MCP client.
    It is intended to be used as a base class for MCP clients that
    use a specific transport (e.g., STDIO, SSE, etc.).
    """

    def __init__(self) -> None:
        """Initialize the MCP client."""
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._tools_list: list[MCPTool] | None = None
        self.stderr: TextIO = sys.stderr

    @abstractmethod
    def create_context_manager(
        self,
        process: Optional[Process] = None,
        stderr: TextIO = sys.stderr,
    ):
        """Create the context manager for the server."""
        pass

    async def connect(self):
        """Connect to the server.

        Connect to the server using ClientSession.
        """
        try:
            transport, process = await self.exit_stack.enter_async_context(
                self.create_context_manager(self._process, stderr=self.stderr)
            )
            read, write = transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
        except Exception as e:
            logger.error(f"Error initializing MCP server: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Cleanup the server connection."""
        async with self._cleanup_lock:
            try:
                if self.exit_stack:
                    await self.exit_stack.aclose()
                await asyncio.get_running_loop().shutdown_asyncgens()
                self.session = None
            except Exception as e:
                logger.error(f"Error cleaning up server: {e}")

    async def list_tools(self) -> list[MCPTool]:
        """List the MCP tools available on the server.

        Returns:
            A list of `MCPTool` objects.
        """
        if not self.session:
            raise RuntimeError(
                "Server not initialized. Make sure you call `connect()` first."
            )
        self._tools_list = (await self.session.list_tools()).tools
        return self._tools_list

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult:
        """Call an MCP tool on the server.

        Args:
            tool_name: The name of the MCP tool to invoke.
            arguments: The arguments to pass to the MCP tool.

        Returns:
            The result of the tool call.
        """
        if not self.session:
            raise RuntimeError(
                "Server not initialized. Make sure you call `connect()` first."
            )
        return await self.session.call_tool(tool_name, arguments)


    async def _invoke_mcp_tool(self, tool: MCPTool, input_json: str) -> str:
        """Invoke an MCP tool and return the result as a string.

        Args:
            tool: The MCP tool to invoke
            input_json: The input to pass to the MCP tool

        Returns:
            The result of the tool call
        """

        try:
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception as e:
            raise RuntimeError(
                f"Invalid JSON input for MCP tool {tool.name}: {input_json}"
            ) from e

        try:
            call_tool_result = await self.call_tool(tool.name, json_data)
            self._log_mcp_tool_call(tool.name, json_data, call_tool_result)
        except Exception as e:
            raise RuntimeError(f"Error invoking MCP tool {tool.name}") from e

        if len(call_tool_result.content) == 1:
            tool_output_item = call_tool_result.content[0]
            return tool_output_item.model_dump_json()

        if len(call_tool_result.content) > 1:
            tool_output_list = [item.model_dump() for item in call_tool_result.content]
            return json.dumps(tool_output_list)

        return "Error invoking MCP tool; call_tool_result.content is an empty list"

    def _log_mcp_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_tool_result: CallToolResult,
    ) -> None:
        """
        Log a MCP tool call.

        Args:
            tool_name: The name of the MCP tool
            tool_args: The input to the MCP tool
            tool_output: The output from the MCP tool
        """

        message_text = (
            f"MCP tool name:\n[bold green]{tool_name}[/bold green]\n\n"
            f"MCP tool args:\n[bold magenta]{tool_args}[/bold magenta]\n\n"
            f"MCP tool call result:\n[bold magenta]{call_tool_result.model_dump_json()}[/bold magenta]"  # noqa: E501
        )
        logger.print(
            message=message_text,
            title="MCP tool call result",
            border_style="blue",
            expand=False,
        )

    async def __aenter__(self):
        """Enter the context manager.

        Upon entering the context manager, we establish a connection to the
        server and return a context manager for the transport.
        """
        logger.debug(f"Entering context manager for {self.name}")
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the context manager.

        Upon exiting the context manager, we close the connection to the server.
        """
        logger.debug(f"Exiting context manager for {self.name}")
        await self.cleanup()


class MCPClientStdio(MCPClientBase):
    """MCP Client that uses the stdio transport to communicate with a MCP Server."""

    def __init__(
        self,
        params: StdioServerParameters,
        name: str | None = None,
    ):
        """
        Create a new MCP Client based on the stdio transport.
        Args:
            params: a dictionary of params that configure the server to be connected to.
            name: A readable name for the client connection.
        """
        super().__init__()
        self.params = params
        self._name = (
            name or f"MCPClientStdio(command={self.params.command}, args={self.params.args})"
        )
        self._process: Optional[Process] = None

    @asynccontextmanager
    async def create_context_manager(
        self,
        process: Optional[Process] = None,
        stderr: TextIO = sys.stderr,
    ):
        """Create the context manager for the server."""
        if not process:
            self._process = await asyncio.create_subprocess_exec(
                self.params.command,
                *self.params.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
            )
            process = self._process
        logger.info(f"MCP client created with pid: {process.pid}")

        async with stdio_client(server=self.params, errlog=stderr) as context_manager:
            yield context_manager, process

    async def cleanup(self):
        """Cleanup the server connection."""
        async with self._cleanup_lock:
            try:
                # Finally, clean the exit stack
                if self.exit_stack:
                    await self.exit_stack.aclose()
                # Still active process, attempt terminating
                if self._process and self._process.returncode is None:
                    try:
                        logger.info(f"Closing MCP process with pid: {self._process.pid}")
                        self._process.terminate()
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)

                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.warning("MCP process was canceled or timed out terminating, attempting to kill")
                        self._process.kill()
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    finally:
                        logger.info(
                            f"MCP process pid: {self._process.pid} returned code {self._process.returncode}"
                        )

                self.session = None
            except Exception as e:
                logger.error(f"Error cleaning up server: {e}")

    @property
    def name(self) -> str:
        """A readable name for the client connection."""
        return self._name


class MCPClientStreamableHttp(MCPClientBase):
    """MCP Client that uses the streamable HTTP transport to communicate with a MCP Server.
    Implements the Streamable HTTP transport as defined in the Model Context Protocol specification.
    See: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http
    """

    def __init__(
        self,
        params: StreamableHttpParameters,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5
    ):
        """Create a new MCP Client based on the HTTP transport.

        Args:
            params: The parameters for the MCP Server.
            name: A readable name for the client connection.
                If not provided, we'll create one from the URL.
            client_session_timeout_seconds: The read timeout passed to the MCP ClientSession.
                Defaults to 5 seconds.
        """
        super().__init__()
        self.params = params
        self._name = name or f"MCPServerStreamableHttp(url={self.params.url})"
        self.client_session_timeout_seconds = client_session_timeout_seconds
        self._process = None  # Not used for HTTP, but required for interface compatibility

    @asynccontextmanager
    async def create_context_manager(
        self,
        process,
        stderr,
    ):
        """Create the context manager for the server using streamable HTTP transport."""
        transport = streamablehttp_client(
            url=self.params.url,
            headers=self.params.headers,
            timeout=self.params.timeout,
            sse_read_timeout=self.params.sse_read_timeout,
            terminate_on_close=self.params.terminate_on_close
        )
        async with transport as (read_stream, write_stream, _):
            yield (read_stream, write_stream), process

    @property
    def name(self) -> str:
        """A readable name for the client connection."""
        return self._name


async def main():
    from clients import MCPClientStreamableHttp
    from mcp.client.session_group import StreamableHttpParameters
    async with MCPClientStreamableHttp(
            params=StreamableHttpParameters(
                url="http://192.168.49.2:31996/mcp",
                headers={"X-MCP-Session-ID": "sandbox-jj68m5fiuypr-47143dd7-8a3c-4104-a5ba-43f47af1af23"}
            ),
            name="Streamable HTTP Python Server",
        ) as mcp_client:
        await mcp_client.connect()
        res = await mcp_client.list_tools()
        for tool in res:
            print(tool.name)

        res = await mcp_client.call_tool("remote_google-search", {"query": "deepseek"})
        for result in json.loads(res.content[0].text)['results']:
            print(result)

        res = await mcp_client.call_tool("terminal-controller_execute_command", {"command": "apt-get update && apt-get install python3"})
        print(res)
        # for line in json.loads(res.content[0].text)['stdout']:
            # print(line)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())