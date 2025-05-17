from typing import Dict, List, Any, Optional
import json
import re
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
import structlog
from agent.common.llms import get_reflection_model
from fastmcp.tools import Tool as FastMCPTool
from agent.common.llm_guard import sanitize_json_output

logger = structlog.get_logger()

def format_messages_as_json(messages: List[BaseMessage]) -> str:
    """
    Format message history as a compact JSON array.
    
    Args:
        messages: List of messages to format
        
    Returns:
        A JSON string representing the messages
    """
    formatted_messages = []
    
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else msg.type
        formatted_messages.append({"role": role, "content": msg.content})
    
    return json.dumps(formatted_messages, ensure_ascii=False)

async def select_relevant_tools(
    messages: List[BaseMessage],
    available_tools: List[FastMCPTool],
    max_tools: int = 15
) -> List[FastMCPTool]:
    """
    Select relevant tools based on the conversation history and user intent.
    
    Args:
        messages: The conversation history between user and assistant
        available_tools: List of all available MCP tools
        max_tools: Maximum number of tools to select (default: 15)
        
    Returns:
        List of selected MCP tools relevant to the conversation
    """
    log = logger.bind()
    
    if not messages:
        log.info("No messages provided, returning all tools")
        return available_tools
    
    if not available_tools:
        log.info("No tools available")
        return []
    
    # Get the reflection model
    reflection_model = get_reflection_model()
    
    # Create a summary of available tools
    tool_descriptions = [{"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema} for tool in available_tools]
    
    # We only include the last 5 messages to keep the context manageable
    recent_messages = messages[-5:]
    
    # Format messages as JSON
    messages_json = format_messages_as_json(recent_messages)
    
    # Create the prompt for tool selection
    prompt = f"""You are an AI assistant tasked with selecting the most relevant tools for a conversation based on user intent.
Given the conversation history below and a list of available tools, create a structured plan and select the tools that are most likely needed to fulfill the user's request.

Conversation History:
{messages_json}

Available Tools:
{json.dumps(tool_descriptions, ensure_ascii=False)}

ANALYSIS INSTRUCTIONS:
1. Analyze the user's most recent message to understand their core intent and needs.
2. Break down the user's request into specific tasks that need to be completed (task decomposition).
3. For each task, identify the specific tools needed to complete it successfully.
4. Consider potential edge cases or alternative approaches that might require additional tools.

TOOL SELECTION INSTRUCTIONS:
1. Select ALL tools that are DIRECTLY relevant to accomplishing ANY part of the user's request.
2. For complex tasks, make sure to include ALL required tools for the complete workflow:
   - Tools for retrieving existing content/data
   - Tools for creating/modifying content
   - Tools for accessing or generating necessary data
   - Tools for specialized operations (search, transformation, analysis)
3. Include tools for potential subtasks that may not be explicitly mentioned but are necessary to complete the request.
4. When working with external services or APIs, include tools for both fetching data AND modifying/creating content.
5. For tasks involving spatial or location data, include relevant geospatial tools.
6. For content creation tasks, ensure tools for both creating containers and adding content are included.

Return your response in the following JSON format. ONLY return this JSON format and nothing else:
{{
    "task_plan": "Brief description of how you plan to approach the user's request, breaking it down into steps",
    "tool_names": ["tool_name_1", "tool_name_2", ...]
}}

Result JSON:
"""
    
    try:
        # Call the reflection model to select tools
        response = await reflection_model.ainvoke(prompt)
        response_text = response.content
        
        # Use sanitize_json_output to parse and extract the JSON
        try:
            tool_selection = sanitize_json_output(response_text)
        except Exception as e:
            log.error(f"Failed to parse JSON response: {e}", response=response_text)
            return available_tools
        
        if not isinstance(tool_selection, dict) or "tool_names" not in tool_selection:
            log.error("Invalid tool selection response format", response=response_text)
            return available_tools
        
        # Get the selected tool names
        selected_tool_names = tool_selection["tool_names"]
        
        if not isinstance(selected_tool_names, list):
            log.error("Invalid tool_names format, expected list", response=response_text)
            return available_tools
        
        # Log the task plan if available
        if "task_plan" in tool_selection and isinstance(tool_selection["task_plan"], str):
            log.info(f"Task plan: {tool_selection['task_plan']}")
        
        # Limit to max_tools
        if len(selected_tool_names) > max_tools:
            selected_tool_names = selected_tool_names[:max_tools]
            log.info(f"Limited selected tools to {max_tools}")
        
        # Filter available tools based on selected names
        selected_tools = [
            tool for tool in available_tools 
            if tool.name in selected_tool_names
        ]
        
        log.info(
            f"Selected {len(selected_tools)}/{len(available_tools)} tools based on conversation",
            selected_tool_names=[tool.name for tool in selected_tools]
        )
        
        # If no tools were selected, return all tools
        if not selected_tools:
            log.warning("No tools matched the selection criteria, returning all tools")
            return available_tools
        
        return selected_tools
    
    except Exception as e:
        log.error(f"Error selecting tools", error=str(e))
        # Fall back to returning all tools
        return available_tools 