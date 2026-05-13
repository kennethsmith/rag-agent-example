import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def llm():
    #return ollama()
    return lm_studio()

def ollama():
    model = 'mistral'
    return ChatOllama(
        model=model,
        temperature=0
    )

def lm_studio():
    model = "llama-3.2-3b-instruct"
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url="http://localhost:1234/v1",
        api_key=os.environ.get("OPENAI_API_KEY") or "<OPENAI_API_KEY>" # LM Studio doesn't require a real key
    )