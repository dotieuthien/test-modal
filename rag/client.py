import json
import requests


class VRAGClient:
    def __init__(self, base_url="https://dotieuthien--example-vision-rag-serve.modal.run"):
        self.base_url = base_url

    def stream_query(self, query: str, doc_id: str):
        """
        Stream results from a VRAG query
        """
        url = f"{self.base_url}/search"
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json'
        }

        # Make the request
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

        print("\n🤖: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                # Decode the line and remove 'data: ' prefix
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        # Parse the JSON chunk
                        chunk_data = json.loads(line[6:])
                        if 'chunk' in chunk_data:
                            print(chunk_data['chunk'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        print()  # New line at the end


def main():
    client = VRAGClient()

    # Example usage
    query = "What is key insights of the report and next steps?"
    doc_id = "999dc499-3dc0-4fa0-9fdc-886a69312edb"
    print(f"\nQuerying: {query}")
    print("-" * 50)

    try:
        client.stream_query(query, doc_id)
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
