"""This simple script shows how to interact with an OpenAI-compatible server from a client."""

import argparse
import time
import base64
import io
import modal
from openai import OpenAI
from PIL import Image


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


def get_completion(client, model_id, messages, args):
    completion_args = {
        "model": model_id,
        "messages": messages,
        "frequency_penalty": args.frequency_penalty,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
        "stop": args.stop,
        "stream": args.stream,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    completion_args = {
        k: v for k, v in completion_args.items() if v is not None
    }

    try:
        response = client.chat.completions.create(**completion_args)
        return response
    except Exception as e:
        print(Colors.RED, f"Error during API call: {e}", Colors.END, sep="")
        return None


def main():
    parser = argparse.ArgumentParser(description="OpenAI Client CLI")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use for completion, defaults to the first available model",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="The workspace where the LLM server app is hosted, defaults to your current Modal workspace",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="The environment in your Modal workspace where the LLM server app is hosted, defaults to your current environment",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="example-sglang-openai-compatible",
        help="A Modal App serving an OpenAI-compatible API",
    )
    parser.add_argument(
        "--function-name",
        type=str,
        default="serve",
        help="A Modal Function serving an OpenAI-compatible API. Append `-dev` to use a `modal serve`d Function.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="super-secret-token",
        help="The API key to use for authentication, set in your api.py",
    )

    # Completion parameters
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--frequency-penalty", type=float, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate. Streaming and chat mode only support n=1.",
    )
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Prompting
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's in this image?",
        help="The user prompt for the chat completion",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a reader and key-value extractor",
        help="The system prompt for the chat completion",
    )

    # UI options
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming of response chunks",
    )
    parser.add_argument(
        "--chat", action="store_true", help="Enable interactive chat mode"
    )

    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    workspace = args.workspace or modal.config._profile

    environment = args.environment or modal.config.config["environment"]

    prefix = workspace + (f"-{environment}" if environment else "")

    client.base_url = (
        f"https://{prefix}--{args.app_name}-{args.function_name}.modal.run/v1"
    )

    if args.model:
        model_id = args.model
        print(
            Colors.BOLD,
            f"🧠: Using model {model_id}. This may trigger a model load on first call!",
            Colors.END,
            sep="",
        )
    else:
        print(
            Colors.BOLD,
            f"🔎: Looking up available models on server at {client.base_url}. This may trigger a model load!",
            Colors.END,
            sep="",
        )
        model = client.models.list().data[0]
        model_id = model.id
        print(
            Colors.BOLD,
            f"🧠: Using {model_id}",
            Colors.END,
            sep="",
        )

    messages = [
        {
            "role": "system",
            "content": args.system_prompt,
        }
    ]

    print(
        Colors.BOLD
        + "🧠: Using system prompt: "
        + args.system_prompt
        + Colors.END
    )

    if args.chat:
        print(
            Colors.GREEN
            + Colors.BOLD
            + "\nEntering chat mode. Type 'bye' to end the conversation."
            + Colors.END
        )
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["bye"]:
                break

            MAX_HISTORY = 10
            if len(messages) > MAX_HISTORY:
                messages = messages[:1] + messages[-MAX_HISTORY + 1 :]
            
            image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )

            response = get_completion(client, model_id, messages, args)

            if response:
                if args.stream:
                    # only stream assuming n=1
                    print(Colors.BLUE + "\n🤖: ", end="")
                    assistant_message = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            print(content, end="")
                            assistant_message += content
                    print(Colors.END)
                else:
                    assistant_message = response.choices[0].message.content
                    print(
                        Colors.BLUE + "\n🤖:" + assistant_message + Colors.END,
                        sep="",
                    )

                messages.append(
                    {"role": "assistant", "content": assistant_message}
                )
    else:
        # image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
        image_path = "/Users/dotieuthien/Documents/rnd/test-modal/benchmarks/images/test/3.png"
        image = Image.open(image_path).convert("RGB")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode(
            "utf-8")
        
        import uuid
        request_id = str(uuid.uuid4())
        
        user_input = "Request ID: " + request_id + """
            Return the results in JSON format for:
            {
                "sender_name": "The sender name, not number, empty if not present", 
                "receiver_name": "The receiver name or business name, not number, empty if not present", 
                "sender_bank_account": "The sender's bank account numbers, empty if not present", 
                "receiver_bank_account": "The receiver's bank account numbers, empty if not present", 
                "bank_name": "The name of the bank or service provider, empty if not present",
                "transaction_id" :"The transaction ID, empty if not present",
                "value": "The transaction amount", 
                "type": "The transaction type: expense when send money to others or income when receive money from others, always return income or expense", 
                "category": "The transaction purpose, categorized as bills, entertainment, education, shopping, others if cannot be categorized, always return value", 
                "time": "The transaction timestamp", 
                "noted": "Notes or messages in the transaction"
            } 
            """
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        )
        print(Colors.GREEN + f"\nYou: {args.prompt}" + Colors.END)
        response = get_completion(client, model_id, messages, args)
        if response:
            if args.stream:
                print(Colors.BLUE + "\n🤖:", end="")
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="")
                print(Colors.END)
            else:
                # only case where multiple completions are returned
                for i, response in enumerate(response.choices):
                    print(
                        Colors.BLUE
                        + f"\n🤖 Choice {i + 1}:{response.message.content}"
                        + Colors.END,
                        sep="",
                    )


if __name__ == "__main__":
    for i in range(10):
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
