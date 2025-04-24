# Figma MCP Client

A Python client for communicating with the Figma MCP (Multi-Client Protocol) server. This client allows you to connect to Figma and use its tools programmatically.

## Setup

1. Create a `.env` file with your Figma API key:
   ```
   FIGMA_TOKEN=your_figma_api_key_here
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   # or if using uv
   uv pip install -e .
   ```

## Usage

The client connects to the Figma MCP server using stdio communication. Here's a basic example:

```python
import asyncio
from figma_client import FigmaMCPClient

async def run_example():
    client = FigmaMCPClient()
    try:
        # Connect to the server and get available tools
        tools = await client.connect_to_server()
        print("Available tools:", tools)
        
        # Example: Execute a tool
        result = await client.execute_tool("tool_name", {"param1": "value1"})
        print("Result:", result)
        
    finally:
        # Always close the client properly
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_example())
```

## Available Tools

After connecting to the server, the client will print a list of available tools. You can execute any of these tools using the `execute_tool` method with appropriate parameters.

## Configuration

The client is configured to use the Figma MCP server with the following settings:
```json
{
  "mcpServers": {
    "Framelink Figma MCP": {
      "command": "npx",
      "args": ["-y", "figma-developer-mcp", "--figma-api-key=YOUR-KEY", "--stdio"]
    }
  }
}
```
