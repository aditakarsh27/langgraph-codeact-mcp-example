import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
REFLECTION_LLM_PROVIDER = os.getenv("REFLECTION_LLM_PROVIDER", "anthropic")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Controls whether tools are filtered based on conversation context
# Currently overfitting to the last user message and doesn't think through - hence disabled
ENABLE_TOOL_FILTERING = os.getenv("ENABLE_TOOL_FILTERING", "false").lower() == "true"
