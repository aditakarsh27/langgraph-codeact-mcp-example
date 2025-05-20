# 🚀 LangGraph CodeAct MCP Example

A powerful, cost-efficient virtual assistant built with LangGraph, LangChain, and Model Context Protocol (MCP) that demonstrates how to efficiently process large tasks through code generation.

[Video demonstration link will be added soon]

## Key Benefits

- **LLM Cost Reduction** - CodeAct approach generates and executes code instead of calling tools one by one, dramatically reducing token usage
- **Reliable Task Execution** - Handles complex tasks requiring hundreds of actions with minimal supervision
- **Context Persistence** - Maintains variable state between interactions, enabling multi-step processes without regenerating data
- **Safe Code Execution** - Uses PyodideSandbox for secure Python execution in a controlled environment

## Project Setup

Clone the repository:

```bash
git clone https://github.com/n-sviridenko/langgraph-codeact-mcp-example
cd langgraph-codeact-mcp-example
```

Install dependencies:

```bash
poetry install
```

Create a `.env` file and update with your API keys:

```bash
cp .env.example .env
```

Run the application:

```bash
# Start MCP server
poetry run python -m dotenv run python server.py

# Start LangGraph server
poetry run python -m dotenv run langgraph dev
```

## Architecture Overview

The system consists of two main components:

1. **Composite MCP Server** - Aggregates multiple MCP services through a unified SSE endpoint
2. **Virtual Assistant Graph** - Implements both CodeAct and ReAct approaches with dynamic selection

### CodeAct vs ReAct Approach

- **ReAct Approach** - Traditional agent pattern that calls tools one by one
  - Higher cost due to growing context window
  - Can be unreliable for large tasks due to potential early termination

- **CodeAct Approach** - Agent generates Python code that executes tools programmatically
  - Dramatically reduces token usage by replacing multiple tool calls with code
  - More deterministic behavior with loops and error handling
  - Maintains variable state between executions
  - Uses Claude for reflection and code validation

### System Components

- **MCP Server** - Composite FastMCP server that mounts multiple services under a single endpoint
  - Exposes tools through Server-Sent Events (SSE) protocol
  - Enables secure access from PyodideSandbox environment

- **Code Execution Environment** - PyodideSandbox WebAssembly-based Python runtime
  - Maintains variable state using thread-based sessions
  - Auto-generates Python function bindings for all MCP tools

- **Virtual Assistant Graph**
  - Dynamically selects between CodeAct and ReAct based on task
  - Optional tool filtering for specific tasks
  - Pre-configured with Notion API and Google Maps services

## Extending the System

Add new MCP services to the `services` dictionary in `setup_mcp_proxy_servers` function:

```python
services = {
    # ... existing services
    "new_service": {
        "package": "@package/new-service-mcp",
        "args": [],
        "env": {
            "NEW_SERVICE_API_KEY": os.getenv("NEW_SERVICE_API_KEY", "")
        }
    }
}
```

## Limitations

- **Container Persistence** - Session data is lost when containers are restarted in cloud environments
- **Potential Hallucinations** - LLMs may occasionally print full data structures despite prompting
- **Rate Limiting** - External API constraints may require custom handling for large datasets
