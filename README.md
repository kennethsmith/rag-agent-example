# Project Title

Simple overview of use/purpose.

## Description

Learning langchain/langgraph and used a notebook (https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb) to adopt to a command line python application.

The plan is to change the the flow of the original notebook, then start to use coding agents to visualize and modify iteratively.

This application is meant to be able to run completely local. It is setup to use Ollama or LM Studio, I'll add more context in the notes section about how to do that.

Next steps will be some fun with opencode and other coding agents to get the codebase in an actual good state.

## Notes
This will run with either Ollam or LM Studio, ./llms/get.py returns the instance of the LLM to the agents. Currently the LLM Server and models are hard coded and require Ollama or LM Studio to be setup with those models.

Running ./artifacts/pull.py will download the embedding, a rlm/rag-prompt from langsmith and some posts on the web. This may require setting up environment variables (I have been able to run without these set):
    LANGSMITH_API_KEY
    NOMIC_API_KEY
    USER_AGENT
The files will be stored in your home directory (~/temp/rag-example/*) and once they are downloaded the application runs completely local.

In order for the web search (./agents/web_search.py) flow to work, you'll need to set the environment variable:
    TAVILY_API_KEY

Running the ./draw.py will create a diagram of the graph (./diagram.png).