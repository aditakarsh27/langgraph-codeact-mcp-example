from fastmcp.tools import Tool as FastMCPTool
from fastmcp.client import Client
from langchain_core.tools import BaseTool, StructuredTool
import json
from typing import Literal

def convert_fastmcp_tool_to_langchain_tool(mcp_tool: FastMCPTool, mcp_client: Client, approach: Literal["react", "codeact"]) -> BaseTool:
    """
    Convert a FastMCP tool to a LangChain BaseTool.
    
    Args:
        mcp_tool: FastMCP Tool instance
        
    Returns:
        A LangChain BaseTool
    """
    async def _tool_func(**kwargs):
        """Function that calls the FastMCP tool."""
        return await mcp_client.call_tool(mcp_tool.name, kwargs)
    
    # Get parameters schema from the tool
    parameters = mcp_tool.inputSchema
    
    # Patch the schema for notion_API-post-page tool
    if mcp_tool.name == "notion_API-post-page":
        # https://github.com/makenotion/notion-mcp-server/issues/49
        if "properties" in parameters and "parent" in parameters["properties"]:
            # Add database_id option and make both optional
            parent_props = parameters["properties"]["parent"]["properties"]
            if "page_id" in parent_props:
                parent_props["database_id"] = {"type": "string", "format": "uuid"}
                
                # Make both optional by removing from required
                if "required" in parameters["properties"]["parent"]:
                    parameters["properties"]["parent"]["required"] = []
                
                # Update description
                mcp_tool.description += "\nNote: Either parent.page_id or parent.database_id must be set."
    
    # Langgraph-codeact doesn't support input schema, so we need to add it to the description
    # Add input schema to description
    enhanced_description = f"{mcp_tool.description}\n\nInput Schema: {json.dumps(parameters)}" if approach == "codeact" else mcp_tool.description
    
    # Create a structured tool with the same name, description, and parameters
    tool = StructuredTool.from_function(
        coroutine=_tool_func,
        name=mcp_tool.name,
        description=enhanced_description,
        args_schema=parameters
    )
    
    return tool
