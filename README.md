# Virtual Assistant Agent

A modular, extensible virtual assistant built with Python, LangGraph, and LangChain that implements the Model Context Protocol (MCP) for managing and executing tasks through specialized agents.

## Architecture Overview

The Virtual Assistant Agent uses a supervisor-based workflow architecture with two main components:

1. **FastMCP Proxy Server** - Manages multiple MCP services through a unified SSE endpoint
2. **Task Executor Agent** - Handles task execution using appropriate MCP tools

### MCP Server Management

The system uses FastMCP to provide the following capabilities:

- **Composite Server Architecture** - Multiple MCP services are mounted on a single FastMCP server
- **Server-Side Events (SSE) Transport** - All MCP services are exposed through a unified SSE endpoint
- **Dynamic Tool Discovery** - Tools are automatically discovered and converted to LangChain-compatible format
- **Code Execution Environment** - Enables Python code execution in a sandboxed environment with access to MCP tools

### Supported MCP Services

The system comes pre-configured with the following MCP services:

- **Firecrawl** - For web search and data extraction
- **Notion API** - For Notion database and page interactions
- **Google Maps** - For location-based services and mapping

### Task Execution

The Task Executor intelligently processes user requests by:

- Analyzing conversation context to select appropriate approach (CodeAct or React)
- Connecting to the FastMCP proxy server through SSE transport
- Dynamically accessing tools from all mounted MCP services
- Executing tasks using Python code (CodeAct) or structured reasoning (React)
- Generating responses based on tool outputs

## Project Setup

Clone the repository:

```bash
git clone <repository-url>
cd virtual-assistant-agent
```

Install dependencies:

```bash
poetry install
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Update the `.env` file with your configuration values, including:

- API keys for language models
- MCP service API keys (FIRECRAWL_API_KEY, NOTION_TOKEN, GOOGLE_MAPS_API_KEY)
- Agent behavior configurations

### Run the Application

Start the MCP server:

```bash
poetry run python -m dotenv run python server.py
```

Start the LangGraph server:

```bash
poetry run python -m dotenv run langgraph dev
```

## Extending the System

The Virtual Assistant can be extended with new capabilities by adding additional MCP services to the FastMCP composite server. To add a new service:

1. Add the service configuration to the `services` dictionary in `setup_mcp_proxy_servers` function
2. Install any necessary NPM packages for the MCP service
3. Add appropriate environment variables for API keys and configuration

Example of adding a new service:

```python
services = {
    # ... existing services
    "new_service": {
        "script_path": "npx",
        "args": ["-y", "new-service-mcp"],
        "env": {
            "NEW_SERVICE_API_KEY": os.getenv("NEW_SERVICE_API_KEY", "")
        }
    }
}
```

No additional code changes are needed as the system will automatically discover and expose tools from the new service.
