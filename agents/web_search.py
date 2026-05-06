import os
from langchain_tavily import TavilySearch
from langchain_classic.schema import Document


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    tool = TavilySearch(
        max_results=3,
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or "<TAVILY_API_KEY>"
    )

    # Web search
    web_results = ""
    docs = tool.invoke({"query": question})
    if("error" in docs):
        print(docs)

    elif("results" in docs):
        web_results = "\n".join([d["content"] for d in docs["results"]])
        web_results = Document(page_content=web_results)
        print(docs)
        print(web_results)

    return {"documents": web_results, "question": question}

def main():
    web_search({"question": "Tell me about memory usage for RAG models?"})

if __name__ == "__main__":
    main()