# install langchain langsmith
import os
import file_paths as fp
import pickle
from pathlib import Path 
from gpt4all import GPT4All
from langsmith import Client
from langchain_community.document_loaders import WebBaseLoader

def posts():
    # Get the posts and save it to a binary file for local loading.
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    Path(fp.posts_dir).expanduser().mkdir(parents=True, exist_ok=True)
    with open(Path(fp.posts_path).expanduser(), 'wb') as f:
        pickle.dump(docs, f)

def prompts():
    # Get the prompt from langsmith and save it to a binary file for local loading.
    # https://smith.langchain.com/hub/rlm/rag-prompt
    '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
    '''
    
    prompt = Client(
        # api_key=os.getenv("LANGSMITH_API_KEY") or "<LANGSMITH_API_KEY>"
    ).pull_prompt("rlm/rag-prompt")
    Path(fp.prompts_dir).expanduser().mkdir(parents=True, exist_ok=True)
    with open(Path(fp.prompts_path).expanduser(), 'wb') as f:
        pickle.dump(prompt, f)

def embeddings():
    # This will download to ~/.cache/gpt4all/ if not present for local loading.
    GPT4All(model_name="nomic-embed-text-v1.5.f16.gguf", allow_download=True)

if __name__ == "__main__":
    posts()
    prompts()
    embeddings()