from typing import Dict, List, Any, Optional, TypedDict, Literal
import structlog
import json
from langgraph.graph import StateGraph, MessagesState
from agent.common.llms import get_react_agent_model, get_reflection_model
from langgraph_codeact import create_codeact
from langchain_core.tools import tool as create_tool
from agent.virtual_assistant.create_pyodide_eval_fn import create_pyodide_eval_fn, make_safe_function_name
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from fastmcp import FastMCP, Client
from fastmcp.client.transports import NpxStdioTransport, SSETransport
from fastmcp.tools import Tool as FastMCPTool
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
import inspect
from agent.common.config import LLM_PROVIDER, ENABLE_TOOL_FILTERING
from agent.virtual_assistant.convert_fastmcp_tool_to_langchain_tool import convert_fastmcp_tool_to_langchain_tool
from agent.virtual_assistant.create_default_prompt import create_default_prompt
from agent.virtual_assistant.create_reflection_prompt import create_reflection_prompt
from agent.virtual_assistant.tool_selection import select_relevant_tools

logger = structlog.get_logger()
model = get_react_agent_model()
reflection_model = get_reflection_model()

class State(MessagesState):
    """State for the virtual assistant graph"""
    approach: Literal['codeact', 'react'] = Field(
        description="The approach to use, either 'codeact' or 'react'",
        default='codeact'
    )

def create_virtual_assistant_graph():
    """
    Create a virtual assistant graph.
    
    Yields:
        A compiled LangGraph within a context manager
    """
    log = logger.bind()
    log.info("Initializing virtual assistant graph")
    
    # Define the node function that runs the appropriate inner graph based on the approach
    async def run_agent(state: State, config: RunnableConfig):
        log = logger.bind()
        approach = state.get('approach', 'codeact')
        thread_id = config.get("metadata", {}).get("thread_id")
        
        log.info(f"Running agent with approach: {approach}, thread_id: {thread_id}")
        
        # Set up SSE transport and connect to MCP client
        sse_url = f"http://127.0.0.1:8000/sse"
        transport = SSETransport(sse_url)
        
        # Connect to the proxy server
        async with Client(transport=transport) as mcp_client:
            # Get available MCP tools as FastMCP Tool objects
            mcp_tools = await mcp_client.list_tools()
            log.info(f"Found {len(mcp_tools)} FastMCP tools")
            
            # Determine whether to filter tools based on configuration
            if ENABLE_TOOL_FILTERING:
                # Filter MCP tools based on conversation history and user intent
                # This uses the reflection model to analyze the user's request and select only the
                # most relevant tools, which helps improve performance and reduce complexity
                filtered_mcp_tools = await select_relevant_tools(
                    messages=state["messages"],
                    available_tools=mcp_tools
                )
                log.info(f"Using {len(filtered_mcp_tools)}/{len(mcp_tools)} MCP tools after filtering")
            else:
                # Use all available tools when filtering is disabled
                filtered_mcp_tools = mcp_tools
                log.info("Tool filtering is disabled. Using all available tools.")
            
            # Convert only the filtered MCP tools to LangChain tools
            langchain_tools = []
            
            for mcp_tool in filtered_mcp_tools:
                try:
                    # Convert to langchain tool
                    langchain_tool = convert_fastmcp_tool_to_langchain_tool(mcp_tool, mcp_client, approach)
                    langchain_tools.append(langchain_tool)
                    log.info(f"Using MCP tool: {mcp_tool.name}")
                except Exception as e:
                    log.error(f"Failed to convert tool {mcp_tool.name}", error=str(e))
            
            log.info(f"Converted {len(langchain_tools)} MCP tools to langchain tools")
            
            if not langchain_tools:
                log.warning("No langchain tools were created from filtered MCP tools")
            
            if approach == 'react':
                # Create React agent
                react_agent = create_react_agent(
                    model=model,
                    tools=langchain_tools,
                )
                inner_graph = react_agent
            else:  # default to codeact
                # Create sandbox eval function with session_id from thread_id
                eval_fn = create_pyodide_eval_fn(
                    # So that it remembers variables from previous runs
                    session_id=thread_id,
                    sse_url=sse_url,
                    mcp_tools=filtered_mcp_tools
                )
                
                # Get reflection prompt
                reflection_prompt = create_reflection_prompt()
                max_reflections = 3  # Default max reflection iterations
                
                # Create CodeAct agent
                inner_graph = create_codeact(
                    model=model,
                    reflection_model=reflection_model,
                    tools=langchain_tools,
                    eval_fn=eval_fn,
                    prompt=create_default_prompt(langchain_tools),
                    reflection_prompt=reflection_prompt,
                    max_reflections=max_reflections
                )
                inner_graph = inner_graph.compile()
            
            # Invoke the inner graph with the current state
            inner_result = await inner_graph.ainvoke(state)
            return inner_result
    
    # Create the StateGraph
    workflow = StateGraph(State)
    
    # Add the agent node and set it as entry point
    workflow.add_node("agent", run_agent)
    workflow.set_entry_point("agent")
    
    # Compile the graph
    return workflow.compile()
