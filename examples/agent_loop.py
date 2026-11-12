"""Give ANY chat model a long-term memory in ~25 lines.

    export VIC_LLM_BASE_URL=https://api.deepseek.com/v1   # or OpenAI, Qwen, Kimi, Ollama...
    export VIC_LLM_API_KEY=sk-...
    export VIC_LLM_MODEL=deepseek-chat
    python examples/agent_loop.py
"""
import json
import os
import urllib.request

from vic_memory import Memory
from vic_memory.llm import from_env

emb, summ = from_env()
mem = Memory(os.path.expanduser("~/.vic_memory.db"), scope="vic", embedder=emb, summarizer=summ)


def chat(messages):
    base = os.environ.get("VIC_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps({"model": os.environ.get("VIC_LLM_MODEL", "gpt-4o-mini"),
                                                  "messages": messages}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ['VIC_LLM_API_KEY']}"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]


print("Chat with memory. Type 'exit' to end the session (memory consolidates on exit).")
while (user := input("you> ").strip()) != "exit":
    mem.observe(user)                                   # 1. perceive
    system = "You are a helpful assistant with long-term memory.\n\n" + mem.context(user)  # 2. remember
    answer = chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    mem.observe(f"Assistant: {answer}")                 # 3. experience
    print("ai>", answer)
mem.flush_working()                                     # 4. end of day
print("sleep:", mem.sleep())                            # 5. sleep consolidates
