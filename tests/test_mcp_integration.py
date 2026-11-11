import pytest
from tools.mcp_client import MCPManager
from google.adk.tools.mcp_tool import McpToolset


def test_mcp_manager_initialization():
    manager = MCPManager()
    assert manager.list_active_toolsets() == []


def test_mcp_stdio_toolset_creation():
    manager = MCPManager()
    # Connect a stdio toolset definition
    toolset = manager.connect_stdio_server(
        server_id="local_sqlite",
        command="python",
        args=["-m", "http.server", "--help"],
    )
    assert isinstance(toolset, McpToolset)
    assert manager.get_toolset("local_sqlite") is toolset
    assert len(manager.list_active_toolsets()) == 1
