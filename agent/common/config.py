import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
REFLECTION_LLM_PROVIDER = os.getenv("REFLECTION_LLM_PROVIDER", "anthropic")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Composio configuration
ENABLE_COMPOSIO_TOOLS = os.getenv("ENABLE_COMPOSIO_TOOLS", "true").lower() == "true"
COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID", "default")
COMPOSIO_TOOLKITS = os.getenv("COMPOSIO_TOOLKITS", "GMAIL").split(",")
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")

# Controls whether tools are filtered based on conversation context
# Currently overfitting to the last user message and doesn't think through - hence disabled
ENABLE_TOOL_FILTERING = False

GRAPH_RECURSION_LIMIT = int(os.environ.get("GRAPH_RECURSION_LIMIT", 25))
