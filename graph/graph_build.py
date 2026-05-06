from graph.graph import GraphState,route,search_web,vectorstore_v_web_search,retrieve,grade_documents,generate,transform_query,decide_to_generate,grade_generation_v_documents_and_question
from langgraph.graph import END, StateGraph, START

def build():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("router", route)
    workflow.add_node("web_search", search_web)
    
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    # workflow.add_edge(START, "retrieve")
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        vectorstore_v_web_search,
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )
    workflow.add_edge("web_search", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()

    return app