from typing import Dict, List, Any, Optional, TypedDict, Literal
import structlog
import json
from langgraph.graph import StateGraph, MessagesState
from agent.common.llms import get_react_agent_model, get_reflection_model
from langgraph_codeact import create_codeact
from langchain_core.tools import tool as create_tool
from agent.virtual_assistant.create_e2b_eval_fn import create_e2b_eval_fn, make_safe_function_name
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
import inspect
from agent.common.config import LLM_PROVIDER, ENABLE_TOOL_FILTERING, GRAPH_RECURSION_LIMIT, COMPOSIO_USER_ID, ENABLE_COMPOSIO_TOOLS, COMPOSIO_TOOLKITS, COMPOSIO_API_KEY
from agent.virtual_assistant.create_default_prompt import create_default_prompt
from agent.virtual_assistant.create_reflection_prompt import create_reflection_prompt
from agent.virtual_assistant.tool_selection import select_relevant_tools

logger = structlog.get_logger()
model = get_react_agent_model()
reflection_model = get_reflection_model()

def create_composio_langchain_tool(composio_tool, composio_client, user_id: str):
    """Create a LangChain tool from a Composio tool."""
    from langchain_core.tools import BaseTool
    from typing import Any, Dict
    
    # Composio tools are dictionaries, not objects
    tool_name = composio_tool.get('name', 'unknown_tool')
    tool_description = composio_tool.get('description', f'Composio tool: {tool_name}')
    
    async def tool_func(**kwargs):
        """Execute the Composio tool."""
        try:
            result = await composio_client.tools.execute(
                user_id=user_id,
                tool_name=tool_name,
                arguments=kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error executing Composio tool {tool_name}: {e}")
            raise
    
    # Create the LangChain tool
    tool = BaseTool(
        name=tool_name,
        description=tool_description,
        func=tool_func,
        coroutine=tool_func
    )
    
    return tool

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
        
        # Get user_id from config or use default from configuration
        user_id = config.get("metadata", {}).get("user_id", COMPOSIO_USER_ID)
        
        # Initialize Composio tools
        langchain_tools = []
        messages = state.get("messages", [])
        search_term = messages[-1].content
        
        if ENABLE_COMPOSIO_TOOLS:
            try:
                from composio import Composio
                from composio_langchain import LangchainProvider
                composio = Composio(provider=LangchainProvider())
                logger.info(f"API key: {COMPOSIO_API_KEY}")
                
                # Check if user needs authentication
                log.info(f"Checking Composio tools for user: {user_id}")
                
                # Try to get available Composio tools
                try:
                    if ENABLE_TOOL_FILTERING:
                        composio_tools = composio.tools.get(user_id=user_id, search=search_term)
                    else:
                        composio_tools = composio.tools.get(user_id=user_id, toolkits=COMPOSIO_TOOLKITS)
                    if composio_tools is None:
                        composio_tools = []
                    log.info(f"Found {len(composio_tools)} Composio tools")
                except Exception as e:
                    log.error(f"Failed to get Composio tools: {e}")
                    log.info("User may need to authenticate with Composio")
                    log.info("To authenticate, visit: https://app.composio.dev/")
                    composio_tools = []
                
                # Debug: Print the first few tools to see their structure
                if composio_tools and len(composio_tools) > 0:
                    log.info(f"First tool structure: {composio_tools[0]}")
                    log.info(f"First tool keys: {list(composio_tools[0].keys()) if isinstance(composio_tools[0], dict) else 'Not a dict'}")
                else:
                    log.info("No Composio tools found - user may need to authenticate")
                    log.info("To authenticate, visit: https://app.composio.dev/")
                
                # Determine whether to filter tools based on configuration
                if ENABLE_TOOL_FILTERING and composio_tools:
                    # Filter Composio tools based on conversation history and user intent
                    filtered_composio_tools = await select_relevant_tools(
                        messages=state["messages"],
                        available_tools=composio_tools
                    )
                    log.info(f"Using {len(filtered_composio_tools)}/{len(composio_tools)} Composio tools after filtering")
                else:
                    # Use all available tools when filtering is disabled
                    filtered_composio_tools = composio_tools
                    log.info("Tool filtering is disabled. Using all available Composio tools.")
                
                langchain_tools = filtered_composio_tools
            except Exception as e:
                log.error(f"Failed to initialize Composio tools: {e}")
        
        # Log the number of tools available
        if not langchain_tools:
            log.warning("No Composio tools were created")
        else:
            log.info(f"Using {len(langchain_tools)} LangChain tools")
        
        # Create the appropriate agent based on approach
        if approach == 'react':
            react_prompt = "You are a helpful assistant."
            react_model = model
            if LLM_PROVIDER == "openai":
                react_prompt += """
Important: Do not ask whether to proceed with using tools - just proceed directly with calling the appropriate tools.
Use the tools that best address the task based on available information. Make reasonable assumptions when needed.
"""
                react_model = model.bind_tools(langchain_tools, parallel_tool_calls=False)
            # Create React agent
            react_agent = create_react_agent(
                model=react_model,
                tools=langchain_tools,
                prompt=react_prompt
            )
            inner_graph = react_agent
        else:  # default to codeact
            # Get user_id from config or use default from configuration
            user_id = config.get("metadata", {}).get("user_id", COMPOSIO_USER_ID)
            
            # Create sandbox eval function with session_id from thread_id
            eval_fn = create_e2b_eval_fn(
                # So that it remembers variables from previous runs
                session_id=thread_id,
                user_id=user_id,
                tools=langchain_tools
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
                prompt=create_default_prompt(langchain_tools, search_term=search_term),
                reflection_prompt=reflection_prompt,
                max_reflections=max_reflections
            )
            inner_graph = inner_graph.compile()
        
        # Invoke the inner graph with the current state
        inner_result = await inner_graph.ainvoke(state, {**config, "recursion_limit": GRAPH_RECURSION_LIMIT})
        return inner_result
    
    # Create the StateGraph
    workflow = StateGraph(State)
    
    # Add the agent node and set it as entry point
    workflow.add_node("agent", run_agent)
    workflow.set_entry_point("agent")
    
    # Compile the graph
    return workflow.compile()
