import logging
from typing import Any, Dict, List, Optional
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages connections to Model Context Protocol (MCP) servers using Google ADK."""

    def __init__(self):
        self._toolsets: Dict[str, McpToolset] = {}

    def connect_stdio_server(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> McpToolset:
        """Connects to a stdio-based MCP server (e.g. npx -y @modelcontextprotocol/server-*)."""
        logger.info(f"Connecting to MCP server '{server_id}' with command: {command} {args or []}")
        
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        connection_params = StdioConnectionParams(server_params=server_params)
        toolset = McpToolset(connection_params=connection_params)
        self._toolsets[server_id] = toolset
        return toolset

    def get_toolset(self, server_id: str) -> Optional[McpToolset]:
        return self._toolsets.get(server_id)

    def list_active_toolsets(self) -> List[McpToolset]:
        return list(self._toolsets.values())
