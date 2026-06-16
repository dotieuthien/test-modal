from io import BytesIO
from typing import Annotated
from pydantic import Field

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context, Image
from mcp.types import Annotations, ImageContent

from image.flux_txt2img import process_flux_txt2img


mcp_server = FastMCP("modal-mcp-tools")
IMAGE_FORMAT = "JPEG"


async def flux_txt2img(prompt: Annotated[str, Field(description="The prompt to generate an image for")], ctx: Context) -> ImageContent:
    """Let's you generate an image using the Flux model."""
    image_bytes = await process_flux_txt2img(prompt)
    image_content = Image(data=image_bytes, format=IMAGE_FORMAT).to_image_content()
    image_content.annotations = Annotations(audience=["user", "assistant"], priority=0.5)
    return image_content


mcp_server.add_tool(flux_txt2img)


if __name__ == "__main__":
    mcp_server.run(transport='stdio')
