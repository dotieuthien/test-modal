import modal

from .graph import edges, nodes, retrieval
from .graph.common import COLOR, PYTHON_VERSION, image


app = modal.App(
    "example-code-langchain",
    image=image,
    secrets=[
        modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"]),
        modal.Secret.from_name(
            "langsmith-secret", required_keys=["LANGCHAIN_API_KEY"]),
    ],
)


def create_sandbox(app) -> modal.Sandbox:
    agent_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        "torch==2.5.0",
        "transformers==4.46.0",
        "huggingface-hub==0.26.0",
        "diffusers",
    )

    return modal.Sandbox.create(
        image=agent_image,
        timeout=60 * 10,  # 10 minutes
        app=app,
        # Modal sandboxes support GPUs!
        gpu="T4",
        # you can also pass secrets here -- note that the main app's secrets are not shared
    )


def run(code: str, sb: modal.Sandbox) -> tuple[str, str]:
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{code}{COLOR['ENDC']}",
        sep="\n",
    )

    exc = sb.exec("python", "-c", code)
    exc.wait()

    stdout = exc.stdout.read()
    stderr = exc.stderr.read()

    if exc.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: Failed with exitcode {sb.returncode}{COLOR['ENDC']}"
        )

    return stdout, stderr


def construct_graph(sandbox: modal.Sandbox, debug: bool = False):
    from langgraph.graph import StateGraph

    from .graph.common import GraphState

    # Crawl the transformers documentation to inform our code generation
    context = retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # Attach our nodes to the graph
    graph_nodes = nodes.Nodes(context, sandbox, run, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # Construct the graph by adding edges
    graph = edges.enrich(graph)

    # Set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph


DEFAULT_QUESTION = "hi there, tell me what is in this figma https://www.figma.com/design/VU3riqaCfa1DUfoCjSwCTH/Test?node-id=0-1&t=XeAMWGTreiscR2t8-1?"


@app.function()
def go(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """Compiles the Python code generation agent graph and runs it, returning the result."""
    sb = create_sandbox(app)

    graph = construct_graph(sb, debug=debug)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        config={"recursion_limit": 50},
    )

    sb.terminate()

    return result["keys"]["response"]


@app.local_entrypoint()
def main(
    question: str = DEFAULT_QUESTION,
    debug: bool = False,
):
    """Sends a question to the Python code generation agent.

    Switch to debug mode for shorter context and smaller model."""
    if debug:
        if question == DEFAULT_QUESTION:
            question = "hi there, how are you?"

    # Import necessary modules
    import sys
    import asyncio
    from pathlib import Path

    # Set up path for importing MCP client
    BASE = Path(__file__).parent
    sys.path.append(str(BASE))
    from mcp_client.figma_client import FigmaMCPClient

    # Get Figma design info from MCP
    async def get_figma_design():
        client = FigmaMCPClient()
        try:
            await client.connect_to_server()
            figma_query = "Analyze the python code for image generation, just need to describe the flow in this Figma design: https://www.figma.com/design/VU3riqaCfa1DUfoCjSwCTH/Test?node-id=0-1&t=XeAMWGTreiscR2t8-1?"
            figma_info = await client.process_query(figma_query)
            print("--------------------------------")
            print(figma_info)
            print("--------------------------------")
            print("Figma design information obtained successfully.")
            return figma_info
        finally:
            await client.close()

    # Run MCP client and get Figma design details
    figma_design_info = asyncio.run(get_figma_design())

    # Create a prompt for generating frontend code based on the Figma design
    fe_code_prompt = f"""
    Generate python code that implements the following Figma design:
    
    FIGMA DESIGN DETAILS:
    {figma_design_info}
    """

    print(f"\nGenerating frontend code based on Figma design...")

    # Send the enhanced prompt to our code generation agent
    result = go.remote(fe_code_prompt, debug=debug)
    print("\nResult:")
    print(result)

    return result
