from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool
from agent.common.config import LLM_PROVIDER, ENABLE_COMPOSIO_TOOLS, COMPOSIO_USER_ID, ENABLE_TOOL_FILTERING
from agent.virtual_assistant.create_e2b_eval_fn import make_safe_function_name, create_composio_prompt_functions
from typing import Optional
import inspect

def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None, search_term: str = ""):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.
"""

    # Add OpenAI-specific instructions
    if LLM_PROVIDER == "openai":
        prompt += """
Important: Do not ask whether to proceed with writing code or implementing a solution - just proceed directly.
Write the code that best addresses the task based on available information. Make reasonable assumptions when needed.
"""

    prompt += """
In addition to the Python Standard Library, you can use the following functions:
"""

    # Add LangChain tools
    for tool in tools:
        # Use coroutine if it exists, otherwise use func
        tool_callable = tool.coroutine if hasattr(tool, "coroutine") and tool.coroutine is not None else tool.func
        # Create a safe function name
        safe_name = make_safe_function_name(tool.name)
        # Determine if it's an async function
        is_async = inspect.iscoroutinefunction(tool_callable)
        # Add appropriate function definition
        prompt += f'''
{"async " if is_async else ""}def {safe_name}{str(inspect.signature(tool_callable))}:
    """{tool.description}"""
    ...
'''
    
    # Add Composio tools if enabled
    if ENABLE_COMPOSIO_TOOLS and ENABLE_TOOL_FILTERING:
        composio_functions = create_composio_prompt_functions(COMPOSIO_USER_ID, search_term)
        if composio_functions:
            prompt += f"""

# Composio Tools (authenticated services like Gmail, Notion, GitHub, etc.)
{composio_functions}
"""

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code.

Note: Your code is already running in an async context. Do not use asyncio.run() as it will cause errors. You can directly await async functions.

For multi-line strings, always use triple quotes (\"\"\") even if the string contains \\n as a character combination rather than actual new lines.

Always use print() statements to explore data structures and function outputs. Simply returning values will not display them back to you for inspection. For example, use print(result) instead of just 'result'.

As you don't know the output schema of the additional Python functions you have access to, start from exploring their contents before building a final solution.

Reminder: use Python code snippets to call tools"""
    return prompt
