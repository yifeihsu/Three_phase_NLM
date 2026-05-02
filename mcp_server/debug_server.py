# debug_server_v2.py
import sys
import os
from typing import TypedDict

# This version *does* use the FastMCP library, so it's compatible with the install command.
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    print(
        "FATAL: The 'mcp' library is not installed in this environment.",
        file=sys.stderr
    )
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    sys.exit(1)

# Define the server object that 'fastmcp install' is looking for
mcp = FastMCP("debug-env")

class EnvironmentInfo(TypedDict):
    """Structured output for the environment check."""
    python_executable: str
    python_version: str
    working_directory: str
    sys_path: list[str]

@mcp.tool()
async def check_environment() -> EnvironmentInfo:
    """
    Returns critical information about the Python execution environment.
    This helps debug environment mismatches between local dev and the CLI.
    """
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "sys_path": sys.path,
    }

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()