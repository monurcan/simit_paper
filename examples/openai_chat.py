"""Chat via the official OpenAI SDK, pointed at an mstar server.

    pip install openai
    mstar serve qwen3_omni     # or: mstar serve bagel
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="qwen3_omni",
    messages=[{"role": "user", "content": "Give me one fun fact."}],
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3_omni",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
