from typing import List
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from agents import rewriter, router, web_search, standard, grader

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        llm: The LLM session instance
        retriever: The vector store retriever
        datasource: What data source we are going to use, either vector store or web.
        documents: The documents to load into the store, then used as filtered docs to do the RAG query.
        web_documents: The documents pulled from the web.
        generation: LLM generation
        hallucination_grade: The results of grading for hallucination.
        answer_grade: The results of grading for quality.
    """

    question: str
    llm: BaseChatModel
    retriever: VectorStoreRetriever
    datasource: str
    documents: List[str]
    web_documents: List[str]
    generation: str
    hallucination_grade: dict[str]
    answer_grade: dict[str]


### Nodes

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("--- I am going to rewrite your question in case it is too vague. ---")
    
    question = state["question"]
    llm = state["llm"]

    r = rewriter.get(llm)
    better_question = rewriter.invoke(r, question)
    return {"question": better_question}

def route(state):
    """
    Route the question to the appropriate datasource.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, datasource, that routes the question to the correct datasource.
    """
    print("--- I only have a finite vectorstore that covers LLM topics so I'll route other questions to a web search. ---")
    
    question = state["question"]
    llm = state["llm"]
    
    r = router.get(llm)
    datasource = router.invoke(r, question)
    return datasource

def search_web(state):
    """
    Search the web for a response.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("--- I don't have RAG context in the vectorstore for this question so I will search the web. ---")
    
    question = state["question"]

    web_documents = web_search.web_search({"question": question})
    return web_documents

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("--- I am retrieving relevant documents based on the question. ---")
    
    question = state["question"]
    retriever = state["retriever"]

    documents = retriever._get_relevant_documents(question, run_manager=None)
    return {"documents": documents}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("--- I will now grade the relevance of the retrieval. ---")
    
    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]

    g = grader.get_retrieval_grader(llm)
    filtered_docs = []
    for d in documents:
        score = grader.invoke_retrieval_grader(g, question, d.page_content)
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("--- I will generate an answer from the RAG context. ---")

    question = state["question"]
    documents = state["documents"]
    llm = state["llm"]

    s = standard.get(llm)
    generation = standard.invoke(s, question, documents)
    return {"generation": generation}

def grade_hallucinations(state):
    """
    Determines whether the generation is grounded in the documents.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("--- I am going to add a hallucination grade and context to the graph state. ---")
    
    documents = state["documents"]
    generation = state["generation"]
    llm = state["llm"]

    h = grader.get_hallucination_grader(llm)
    score = grader.invoke_hallucination_grader(h, documents, generation)
    return {"hallucination_grade": {"grade": score["score"], "reason": score["reason"]}}

def grade_answer(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("--- I am going to add an answer grade and context to the graph state. ---")

    question = state["question"]
    generation = state["generation"]
    llm = state["llm"]

    g = grader.get_answer_grader(llm)
    score = grader.invoke_answer_grader(g, question, generation)
    return {"answer_grade": {"grade": score["score"], "reason": score["reason"]}}

def end(state):
    """
    Determines any final state cleanup that may need to happen and returns final state.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The final graph state
    """
    print("--- I am at my end state and will return the full state for this session. ---")
    return state

### Edges

def vectorstore_v_web_search(state):
    """
    Determines whether to use the vectorstore or do a web search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    datasource = state["datasource"]
    if datasource == "vectorstore":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "--- We are going to use the vectorstore. ---"
        )
        return "retrieve"
    else:
        # We have relevant documents, so generate answer
        print("--- We are going to search the web. ---")
        return "web_search"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("--- There are no documents that are related to the query, we'll do a web search. ---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("--- There are documents that are related to the query, we'll generate a response. ---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    
    if state["answer_grade"]["grade"] == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
    
    if state["hallucination_grade"]["grade"] == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "supported"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
