"""
Create a reflection prompt for the CodeAct agent to self-review generated code.
"""
from typing import Optional

def create_reflection_prompt() -> str:
    """
    Creates a reflection prompt for the CodeAct agent to validate its own code.
    
    The prompt instructs the model to check code quality and catch common errors.
    
    Returns:
        str: The reflection prompt text
    """
    return """You are a code quality reviewer examining the conversation between a user and an AI assistant. 
The user evaluates the assistant's code and either returns execution results or points out corrections needed.

Check for these quality issues in code:

1. The code should NOT use asyncio.run(). The code is already running in an async context, 
   so the correct approach is to directly await async functions.

2. The code should use triple quotes (\"\"\") for multi-line strings, not single quotes with \\n.
   This applies even if the string contains \\n as a character combination rather than actual new lines.

3. The code should use print() for outputs that need inspection. 
   Simply returning values will not display them back to the assistant for inspection.

4. Before building a solution, the code should first explore unknown tool outputs
   to understand their schema and content, as the assistant doesn't know the output format in advance.

5. The code should reference existing variables and previously computed data instead of 
   duplicating or recreating large data structures. Specifically, when data is retrieved from 
   tool execution or API calls, DO NOT hardcode those results in subsequent code snippets.
   Instead, refer to the variables that contain the execution results. Reuse what's already available.

6. Remember that although the conversation may contain several separate Python code snippets,
   they share the same execution context. Variables, functions, and imports defined in earlier
   snippets are available to later snippets. Avoid redefining or reimporting what's already available.""" 