import asyncio
import os
import sys
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# Load environment variables (ensure both FIGMA_TOKEN and OPENAI_API_KEY are set)
load_dotenv()
FIGMA_TOKEN = os.getenv("FIGMA_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not FIGMA_TOKEN:
    print("Error: FIGMA_TOKEN environment variable not set.")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)


class FigmaMCPClient:
    def __init__(self):
        """
        Initialize the MCP client with Figma MCP server configuration
        """
        self.command = "npx"
        self.args = ["-y", "figma-developer-mcp",
                     f"--figma-api-key={FIGMA_TOKEN}", "--stdio"]
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def connect_to_server(self):
        """Connect to the Figma MCP server via stdio"""
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env={"FIGMA_TOKEN": FIGMA_TOKEN}
        )

        # Establish stdio transport
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport

        # Create MCP client session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # List and display available tools
        resp = await self.session.list_tools()
        tools = [tool.name for tool in resp.tools]
        print(f"Connected to Figma MCP Server. Available tools: {tools}")
        return tools

    async def close(self):
        """Close the client session and exit stack"""
        await self.exit_stack.aclose()

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        if not self.session:
            raise RuntimeError(
                "Not connected to Figma MCP server. Call connect_to_server() first.")

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # print(messages)
        # print(available_tools)

        # Initial OpenAI API call
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            max_tokens=1000
        )

        # print(response)

        # Process response and handle tool calls
        final_text = []

        assistant_message = response.choices[0].message

        # Add assistant message with tool_calls to the history
        assistant_message_dict = {"role": "assistant"}

        # Add content if it exists (use empty string if None)
        if assistant_message.content:
            final_text.append(assistant_message.content)
            assistant_message_dict["content"] = assistant_message.content
        else:
            # Empty string if no content
            assistant_message_dict["content"] = ""

        # Include tool_calls if present
        if assistant_message.tool_calls:
            assistant_message_dict["tool_calls"] = []
            for tool_call in assistant_message.tool_calls:
                assistant_message_dict["tool_calls"].append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })

        # Add the complete assistant message to history
        messages.append(assistant_message_dict)

        print(f"Assistant message: {assistant_message.content}")
        print(f"Assistant message tool calls: {assistant_message.tool_calls}")

        # Process tool calls if present
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"Tool name: {tool_name}")
                print(f"Tool args: {tool_args}")

                # Convert string arguments to dictionary if needed
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(
                    f"[Calling tool {tool_name} with args {tool_args}]")

                # Add tool results to messages using OpenAI's format
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(result)  # Ensure result is string
                })

                # print(messages)

                # Get next response from OpenAI
                response = await self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=available_tools,
                    max_tokens=1000
                )

                if response.choices[0].message.content:
                    final_text.append(response.choices[0].message.content)

        return "\n".join(final_text)


async def main():
    """Example usage of the Figma MCP client"""
    client = FigmaMCPClient()
    try:
        # Connect to server
        tools = await client.connect_to_server()
        print("Available tools:", tools)

        # Interactive query processing
        while True:
            # Get user input
            query = input("\nEnter your query (or 'exit' to quit): ")

            # Exit condition
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break

            # Process the query
            print("\nProcessing query, please wait...")
            try:
                result = await client.process_query(query)
                print("\nResult:")
                print(result)
            except Exception as e:
                print(f"\nError processing query: {e}")
    finally:
        # Ensure we close the connection
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
