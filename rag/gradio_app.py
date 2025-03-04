import gradio as gr
from client import VRAGClient
import requests
import json


def process_query(query: str, doc_id: str) -> str:
    client = VRAGClient()

    # Initialize an empty string to store the response
    full_response = ""

    try:
        url = f"{client.base_url}/search"
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json'
        }

        # Make the request using the client's configuration
        response = requests.post(
            url,
            json={
                'query': query,
                'instance_id': doc_id,
                'count': 3
            },
            headers=headers,
            stream=True,
            verify=False
        )

        # Process the streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        chunk_data = json.loads(line[6:])
                        if 'chunk' in chunk_data:
                            full_response += chunk_data['chunk']
                            yield full_response
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        return f"Error: {str(e)}"


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vision RAG Query Interface")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your query",
                placeholder="What would you like to know?",
                lines=3
            )
            doc_id_input = gr.Textbox(
                label="Document ID",
                placeholder="Enter the document ID",
                value="999dc499-3dc0-4fa0-9fdc-886a69312edb"  # Default value from example
            )
            submit_btn = gr.Button("Submit Query")

        with gr.Column():
            output = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True
            )

    submit_btn.click(
        fn=process_query,
        inputs=[query_input, doc_id_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.queue().launch()
