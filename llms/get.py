import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def llm():
    return ollama()

def ollama():
    model = 'mistral'
    return ChatOllama(
        model=model,
        temperature=0,
        format="json"
    )

def lm_studio():
    model = "qwen3-8b"
    return ChatOpenAI(
        model=model,  # Found in LM Studio server logs/tab
        temperature=0,
        base_url="http://localhost:1234/v1",
        api_key=os.environ.get("OPENAI_API_KEY") or "<OPENAI_API_KEY>" # LM Studio doesn't require a real key
    )