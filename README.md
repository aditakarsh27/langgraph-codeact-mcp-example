# 🚀 LangGraph CodeAct Composio Example

A powerful, cost-efficient virtual assistant built with LangGraph, LangChain, and Composio that demonstrates how to efficiently process large tasks through code generation.

## Video Overview
[![Video Overview](https://img.youtube.com/vi/e5MV23koc-0/0.jpg)](https://www.youtube.com/watch?v=e5MV23koc-0)

## Key Benefits

- **LLM Cost Reduction** - CodeAct approach generates and executes code instead of calling tools one by one, dramatically reducing token usage
- **Reliable Task Execution** - Handles complex tasks requiring hundreds of actions with minimal supervision
- **Context Persistence** - Maintains variable state between interactions, enabling multi-step processes without regenerating data
- **Safe Code Execution** - Uses E2B Sandbox for secure Python execution in a controlled environment

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
# Start LangGraph server
poetry run python -m dotenv run langgraph dev
```

## Architecture Overview

The system consists of two main components:

1. **Composio Integration** - Provides authenticated access to external services like Gmail, Notion, GitHub, etc.
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

- **Composio Integration** - Provides authenticated access to external services
  - Supports Gmail, Notion, GitHub, Slack, and other services
  - Automatic tool discovery and authentication
  - Seamless integration with E2B Sandbox environment

- **Code Execution Environment** - E2B Sandbox container-based Python runtime
  - Maintains variable state using thread-based sessions
  - Auto-generates Python function bindings for all Composio tools

- **Virtual Assistant Graph**
  - Dynamically selects between CodeAct and ReAct based on task
  - Optional tool filtering for specific tasks
  - **Composio Integration** - Supports authenticated tools from Gmail, Notion, GitHub, and other services

## Extending the System

### Composio Integration

The system uses Composio for authenticated access to external services. To configure Composio tools:

1. **Install Composio**: The project includes Composio dependencies in `pyproject.toml`
2. **Configure Environment**: Set the following environment variables:
   ```bash
   ENABLE_COMPOSIO_TOOLS=true
   COMPOSIO_USER_ID=your_user_id
   COMPOSIO_TOOLKITS=GMAIL,GITHUB,NOTION
   COMPOSIO_SEARCH_TERM=your_search_term
   ```
3. **Authenticate Services**: Use Composio's authentication flow to connect to services like Gmail, Notion, GitHub, etc.
4. **Access Tools**: Composio tools are automatically available as Python functions in the CodeAct environment

Composio tools are automatically converted to Python functions and made available in the E2B Sandbox execution environment, allowing the agent to use authenticated APIs directly in generated code.

### Adding New Services

To add new Composio services, simply update the `COMPOSIO_TOOLKITS` environment variable with the desired service names. Available toolkits include GMAIL, GITHUB, NOTION, SLACK, and others supported by Composio.

## Limitations

- **Container Persistence** - Session data is lost when containers are restarted in cloud environments
- **Potential Hallucinations** - LLMs may occasionally print full data structures despite prompting
- **Rate Limiting** - External API constraints may require custom handling for large datasets
