from typing import Dict, List, Any, Callable, Optional
import inspect
import re
import json
import asyncio
from langchain_core.messages import AIMessage
from langgraph_codeact import EvalCoroutine
from langchain_sandbox import PyodideSandbox
from langchain_core.tools import BaseTool
from fastmcp.tools import Tool as FastMCPTool

sandbox = PyodideSandbox("./sessions", allow_net=True)

def safe_repr(obj):
    """Create a safe representation of objects, handling multiline strings."""
    if isinstance(obj, str) and ('\n' in obj or '\r' in obj):
        # Escape any triple quotes in the string
        escaped = obj.replace("'''", r"\'\'\'")
        return f"'''{escaped}'''"
    elif isinstance(obj, dict):
        items = [f"{repr(k)}: {safe_repr(v)}" for k, v in obj.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(obj, list):
        items = [safe_repr(item) for item in obj]
        return "[" + ", ".join(items) + "]"
    else:
        return repr(obj)

def make_safe_function_name(name: str) -> str:
    """Convert a tool name to a valid Python function name."""
    # Replace non-alphanumeric characters with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure the name doesn't start with a digit
    if safe_name and safe_name[0].isdigit():
        safe_name = f"tool_{safe_name}"
    # Handle empty name edge case
    if not safe_name:
        safe_name = "unnamed_tool"
    return safe_name

def create_pyodide_eval_fn(
    session_id: str | None = None,
    sse_url: str = "http://127.0.0.1:8000/sse",
    mcp_tools: List[FastMCPTool] = None
) -> EvalCoroutine:
    """Create an eval_fn that uses PyodideSandbox.

    Args:
        session_id: ID of the session to use
        sse_url: URL of the FastMCP SSE server
        mcp_tools: List of FastMCP Tool objects

    Returns:
        A function that evaluates code using PyodideSandbox
    """
    mcp_tools = mcp_tools or []
    
    # Get tool names to create safe function names
    tool_names = {make_safe_function_name(tool.name): tool.name for tool in mcp_tools}
    
    async def async_eval_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Create static tool function definitions for all tools
        tool_functions = ""
        for safe_name, original_name in tool_names.items():
            tool_functions += f"""            async def {safe_name}(**kwargs):
                response = await _mcp_client.call_tool("{original_name}", kwargs)
                
                def process_item(item):
                    if hasattr(item, 'text'):
                        try:
                            return json.loads(item.text)
                        except:
                            return item.text
                    return item
                
                if isinstance(response, list):
                    return [process_item(item) for item in response]
                return process_item(response)
            
"""
        
        # Create a wrapper function that will execute the code and return locals
        wrapper_code = f"""
# MCP client initialization
import ssl
import uvicorn
import ssl
import json
import re
from fastmcp import Client, client

async def execute():
    try:
        # Create SSE transport to connect to the proxy server
        _transport = client.transports.SSETransport("{sse_url}")
        
        async with Client(transport=_transport) as _mcp_client:
            # Define all tool functions
{tool_functions}
            # Execute the provided code
{chr(10).join("            " + line for line in code.strip().split(chr(10)))}
            return locals()
    except Exception as e:
        import traceback
        print(f"Error: {{e}}")
        print(traceback.format_exc())
        return {{"error": str(e)}}

await execute()
"""
        # Convert functions in _locals to their string representation
        context_setup = ""
        for key, value in _locals.items():
            if callable(value):
                if key not in tool_names:  # Skip if this is one of our MCP tools
                    # Get the function's source code for non-MCP tools
                    src = inspect.getsource(value)
                    context_setup += f"\n{src}"
            else:
                context_setup += f"\n{key} = {safe_repr(value)}"

        try:
            # Execute the code and get the result
            response = await sandbox.execute(
                code=context_setup + "\n\n" + wrapper_code,
                session_id=session_id,
            )

            # Check if execution was successful
            if response.stderr:
                return f"Error during execution: {response.stderr}", {}

            # Get the output from stdout
            output = (
                response.stdout if response.stdout else "<Code ran, no output printed to stdout>"
            )
            result = response.result

            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                return f"Error during execution: {result['error']}", {}

            # Get the new variables by comparing with original locals
            new_vars = {
                k: v for k, v in result.items() if k not in _locals and not k.startswith("_")
            }
            return output, new_vars

        except Exception as e:
            return f"Error during PyodideSandbox execution: {repr(e)}", {}

    return async_eval_fn
