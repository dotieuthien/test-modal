import openai


client = openai.OpenAI(
    api_key="super-secret-token",             # pass litellm proxy key, if you're using virtual keys
    base_url="https://dotieuthien--example-litellm-proxy-serve.modal.run", # litellm-proxy-base url
)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages = [
        {
            "role": "user",
            "content": "what llm are you?"
        }
    ],
)

response_text = response.choices[0].message.content

print("#"*100)
print(response_text)