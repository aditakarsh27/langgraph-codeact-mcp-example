from typing import Dict, List, Any, Callable, Optional
import inspect
import re
import json
import asyncio
import os
import structlog
import aiofiles
from langchain_core.messages import AIMessage
from langgraph_codeact import EvalCoroutine
from langchain_sandbox import PyodideSandbox
from langchain_core.tools import StructuredTool
from agent.common.config import ENABLE_COMPOSIO_TOOLS, COMPOSIO_USER_ID, COMPOSIO_TOOLKITS, ENABLE_TOOL_FILTERING
# Add Composio imports
try:
    from composio import Composio
    from composio_langchain import LangchainProvider
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("Composio not available. Install with: pip install composio")

logger = structlog.get_logger()

# Ensure sessions directory exists
sessions_dir = "./sessions"
os.makedirs(sessions_dir, exist_ok=True)

# Initialize PyodideSandbox with proper error handling
sandbox = None
try:
    logger.info("Initializing PyodideSandbox...")
    sandbox = PyodideSandbox(sessions_dir, allow_net=True)
    logger.info("PyodideSandbox initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PyodideSandbox: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {str(e)}")
    sandbox = None

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

def create_composio_tool_functions(user_id: str = "default", tools: list[StructuredTool] | None = None, search_term: str = "") -> str:
    """Create Python function definitions for Composio tools."""
    if not COMPOSIO_AVAILABLE:
        return ""
    
    try:
        composio = Composio(provider=LangchainProvider())
        
        # Get tools for the user with specific toolkits and search term
        try:
            if tools is None:
                if ENABLE_TOOL_FILTERING:
                    tools = composio.tools.get(user_id=user_id, search=search_term)
                else:
                    tools = composio.tools.get(user_id=user_id, toolkits=COMPOSIO_TOOLKITS)
                logger.info(f"Found {len(tools) if tools else 0} Composio tools for user {user_id}")
                logger.info(f"Using toolkits: {COMPOSIO_TOOLKITS}")
            
            # Debug: Print the first few tools to see their structure
            if tools and len(tools) > 0:
                logger.info(f"First tool structure: {tools[0]}")
                logger.info(f"First tool keys: {list(tools[0].keys()) if isinstance(tools[0], dict) else 'Not a dict'}")
        except Exception as e:
            logger.error(f"Failed to get Composio tools: {e}")
        
        if not tools:
            logger.info("No Composio tools available")
            return ""
        
        # Create function definitions for each tool
        function_definitions = []
        
        for tool in tools:
            try:
                # Composio tools are structured tools
                tool_name = tool.name
                tool_description = tool.description
                
                # Create a safe function name
                safe_name = make_safe_function_name(tool_name)
                
                # Get tool parameters from the schema
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                # Create parameter string
                param_list = []
                required_params = []
                optional_params = []
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'str')
                    if param_type == 'string':
                        param_type = 'str'
                    elif param_type == 'integer':
                        param_type = 'int'
                    elif param_type == 'boolean':
                        param_type = 'bool'
                    elif param_type == 'array':
                        param_type = 'list'
                    elif param_type == 'object':
                        param_type = 'dict'
                    
                    # Add default value if available
                    default_value = param_info.get('default')
                    if default_value is not None:
                        optional_params.append(f"{param_name}: {param_type} = {safe_repr(default_value)}")
                    else:
                        required_params.append(f"{param_name}: {param_type}")
                
                # Combine required parameters first, then optional parameters
                param_list = required_params + optional_params
                param_str = ", ".join(param_list)
                
                # Create function definition with actual Composio execution
                function_def = f"""
async def {safe_name}({param_str}):
    \"\"\"
    {tool_description}
    
    Parameters:
{chr(10).join(f'    {k}: {v.get("description", "No description")}' for k, v in properties.items()) if properties else '    None'}
    \"\"\"
    try:
        # Prepare arguments for the tool
        args = {{}}
{chr(10).join(f'        args["{param}"] = {param}' for param in properties.keys())}
        
        # Execute the tool via Composio
        result = await composio.tools.execute(
            user_id=user_id,
            tool_name="{tool_name}",
            arguments=args
        )
        print(f"Composio tool {tool_name} executed successfully")
        return result
    except Exception as e:
        print(f"Error executing Composio tool {tool_name}: {{e}}")
        print(f"Tool parameters: {{args if 'args' in locals() else 'Not available'}}")
        raise
"""
                function_definitions.append(function_def)
                logger.info(f"Created function for Composio tool: {tool_name} -> {safe_name}")
                
            except Exception as e:
                logger.error(f"Failed to create function for Composio tool {tool_name}: {e}")
                continue
        
        if function_definitions:
            # Add Composio client initialization
            composio_setup = f"""
# Import Composio
from composio import Composio
from composio_langchain import LangchainProvider

# Initialize Composio client
composio = Composio(provider=LangchainProvider())
user_id = "default"
"""
            
            return composio_setup + "\n".join(function_definitions)
        else:
            return ""
            
    except Exception as e:
        logger.error(f"Failed to initialize Composio tools: {e}")
        logger.error(f"Composio error type: {type(e).__name__}")
        # Return a basic setup that will handle the error gracefully
        return """
# Composio initialization failed - tools will not be available
print("Warning: Composio tools are not available due to initialization error")
composio = None
user_id = "default"
"""

def create_composio_prompt_functions(user_id: str = "default", tools: list[StructuredTool] | None = None, search_term: str = "") -> str:
    """Create Python function definitions for Composio tools to be included in prompts."""
    if not COMPOSIO_AVAILABLE:
        return ""
    
    try:
        composio = Composio(provider=LangchainProvider())
        
        # Get tools for the user with specific toolkits and search term
        try:
            if tools is None:
                if ENABLE_TOOL_FILTERING:
                    tools = composio.tools.get(user_id=user_id, search=search_term)
                else:
                    tools = composio.tools.get(user_id=user_id, toolkits=COMPOSIO_TOOLKITS)
            logger.info(f"Found {len(tools) if tools else 0} Composio tools for prompt generation")
        except Exception as e:
            logger.error(f"Failed to get Composio tools for prompt: {e}")
            try:
                tools = composio.tools.get(user_id=user_id, search=search_term)
                logger.info(f"Found {len(tools) if tools else 0} Composio tools without toolkit filtering")
            except Exception as e2:
                logger.error(f"Failed to get any Composio tools for prompt: {e2}")
                return ""
        
        if not tools:
            return ""
        
        # Create function definitions for each tool (for prompt)
        function_definitions = []
        
        for tool in tools:
            try:
                # Composio tools are dictionaries
                tool_name = tool.name
                tool_description = tool.description
                
                # Create a safe function name
                safe_name = make_safe_function_name(tool_name)
                
                # Get tool parameters from the schema
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                # Create parameter string
                param_list = []
                required_params = []
                optional_params = []
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'str')
                    if param_type == 'string':
                        param_type = 'str'
                    elif param_type == 'integer':
                        param_type = 'int'
                    elif param_type == 'boolean':
                        param_type = 'bool'
                    elif param_type == 'array':
                        param_type = 'list'
                    elif param_type == 'object':
                        param_type = 'dict'
                    
                    # Add default value if available
                    default_value = param_info.get('default')
                    if default_value is not None:
                        optional_params.append(f"{param_name}: {param_type} = {safe_repr(default_value)}")
                    else:
                        required_params.append(f"{param_name}: {param_type}")
                
                # Combine required parameters first, then optional parameters
                param_list = required_params + optional_params
                param_str = ", ".join(param_list)
                
                # Create function definition for prompt (without implementation)
                function_def = f"""
async def {safe_name}({param_str}):
    \"\"\"
    {tool_description}
    
    Parameters:
{chr(10).join(f'    {k}: {v.get("description", "No description")}' for k, v in properties.items()) if properties else '    None'}
    \"\"\"
    # This function is available in the execution environment
    # Implementation is handled by Composio client
    pass
"""
                function_definitions.append(function_def)
                
            except Exception as e:
                logger.error(f"Failed to create prompt function for Composio tool {tool_name}: {e}")
                continue
        
        if function_definitions:
            return "\n".join(function_definitions)
        else:
            return ""
            
    except Exception as e:
        logger.error(f"Failed to create Composio prompt functions: {e}")
        return ""

def create_pyodide_eval_fn(
    session_id: str | None = None,
    user_id: str = "default",
    tools: list[StructuredTool] | None = None
) -> EvalCoroutine:
    """Create an eval_fn that uses PyodideSandbox.

    Args:
        session_id: ID of the session to use
        sse_url: URL for SSE transport
        mcp_tools: List of MCP tools to include
        user_id: User ID for Composio tools

    Returns:
        A function that evaluates code using PyodideSandbox
    """
    
    async def async_eval_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Create a wrapper function that will execute the code and return locals
        wrapper_code = f"""
import json
import re

async def execute():
    try:
        # Execute the provided code
{chr(10).join("        " + line for line in code.strip().split(chr(10)))}
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
                # Skip MCP tool functions, LangChain tool functions, and any function with underscore prefix
                if not key.startswith('_'):
                    try:
                        # Get the function's source code
                        src = inspect.getsource(value)
                        # Additional check: skip if the source contains MCP tool patterns or LangChain tool patterns
                        if ('mcp_client.call_tool' not in src and 
                            'mcp_tool.name' not in src and
                            'execute_tool(tool, kwargs)' not in src and
                            'Wrapper function for composio action' not in src):
                            context_setup += f"\n{src}"
                    except (OSError, TypeError):
                        # Skip functions that can't be inspected (like built-ins or MCP tools)
                        pass
            else:
                context_setup += f"\n{key} = {safe_repr(value)}"

        # Add Composio tools to context_setup if enabled
        if ENABLE_COMPOSIO_TOOLS and COMPOSIO_AVAILABLE:
            composio_functions = create_composio_tool_functions(user_id, tools)
            if composio_functions:
                context_setup += f"\n# Composio Tools\n{composio_functions}"
                logger.info("Added Composio tools to context_setup")
        elif not COMPOSIO_AVAILABLE:
            logger.info("Composio tools disabled - Composio not available")
        elif not ENABLE_COMPOSIO_TOOLS:
            logger.info("Composio tools disabled via configuration")

        try:
            # Check if sandbox is properly initialized
            if sandbox is None:
                logger.error("PyodideSandbox is not initialized")
                return "Error: PyodideSandbox is not properly initialized. Please check your environment setup and ensure langchain-sandbox is properly installed.", {}
            logger.info(f"Executing code with session_id: {session_id}")
            
            # Execute the code and get the result
            
            # Save the code being executed to a file
            full_code = context_setup + "\n\n" + wrapper_code
            if session_id:
                code_file_path = os.path.join(sessions_dir, f"{session_id}_code.py")
            else:
                code_file_path = os.path.join(sessions_dir, "code_execution.py")
            
            try:
                async with aiofiles.open(code_file_path, 'w', encoding='utf-8') as f:
                    await f.write(full_code)
                logger.info(f"Code saved to: {code_file_path}")
            except Exception as e:
                logger.error(f"Failed to save code to file: {e}")
            
            response = await sandbox.execute(
                code=full_code,
                session_id=session_id
            )
            logger.info(f"Response: {response}")

            # Check if execution was successful
            if response.stderr:
                logger.error(f"PyodideSandbox stderr: {response.stderr}")
                return f"Error during execution: {response.stderr}", {}

            # Get the output from stdout
            output = (
                response.stdout if response.stdout else "<Code ran, no output printed to stdout>"
            )
            result = response.result or {}

            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Error in result: {result['error']}")
                return f"Error during execution: {result['error']}", {}

            # Get the new variables by comparing with original locals
            new_vars = {
                k: v for k, v in result.items() if k not in _locals and not k.startswith("_")
            }
            
            logger.info(f"Code execution completed successfully. Output length: {len(output)}")
            return output, new_vars

        except Exception as e:
            logger.error(f"Exception during PyodideSandbox execution: {repr(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            return f"Error during PyodideSandbox execution: {repr(e)}", {}

    return async_eval_fn
