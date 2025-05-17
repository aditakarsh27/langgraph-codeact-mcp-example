from fastmcp import FastMCP, Client
from fastmcp.client.transports import NpxStdioTransport
import structlog
import os
import json
import asyncio
import time

from agent.common.config import (
    FIRECRAWL_API_KEY,
    NOTION_TOKEN,
    GOOGLE_MAPS_API_KEY
)

logger = structlog.get_logger()

def setup_mcp_proxy_servers():
    """
    Setup proxy servers for various MCP services.
    Returns the host and port information for the SSE server.
    """
    log = logger.bind()
    
    # Create a composite FastMCP server
    composite_mcp = FastMCP("CompositeServer")
    
    # Setup function for each service in a thread
    async def _run_proxy():
        # Define services with their configurations
        services = {
            "firecrawl": {
                "package": "firecrawl-mcp",
                "args": [],
                "env": {
                    "FIRECRAWL_API_KEY": FIRECRAWL_API_KEY
                }
            },
            "notion": {
                "package": "@notionhq/notion-mcp-server",
                "args": [],
                "env": {
                    "OPENAPI_MCP_HEADERS": json.dumps({
                        "Authorization": f"Bearer {NOTION_TOKEN}",
                        "Notion-Version": "2022-06-28"
                    })
                }
            },
            "google-maps": {
                "package": "@modelcontextprotocol/server-google-maps",
                "args": [],
                "env": {
                    "GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY
                }
            }
        }
        
        # Initialize each service and mount to the composite server
        for service_name, config in services.items():
            try:
                # Create transport with environment variables
                transport = NpxStdioTransport(
                    package=config["package"],
                    args=config["args"],
                    env_vars=config["env"]
                )
                
                # Create client with explicit transport
                stdio_client = Client(transport)
                
                # Create a proxy server for this service
                service_mcp = FastMCP.from_client(stdio_client, name=f"{service_name.capitalize()}Server")
                
                # Mount the service to the composite server
                composite_mcp.mount(service_name, service_mcp)
                
                log.info(f"Mounted {service_name} MCP service")
            except Exception as e:
                log.error(f"Failed to initialize {service_name} service", error=str(e))
        
        # Run the composite server with SSE transport
        host = "127.0.0.1"
        port = 8000
        await composite_mcp.run_async(transport="sse", host=host, port=port)
    
    # Start the proxy server in a separate thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_proxy())
    
    # Give the server a moment to start
    time.sleep(2)
    
    log.info("MCP proxy server started on http://127.0.0.1:8000/sse")

if __name__ == "__main__":
    setup_mcp_proxy_servers()