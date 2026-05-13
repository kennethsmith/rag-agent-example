from graph.graph import GraphState,route,search_web,vectorstore_v_web_search,retrieve,grade_documents,generate,transform_query,decide_to_generate,grade_generation_v_documents_and_question,end,grade_answer,grade_hallucinations
from langgraph.graph import END, StateGraph, START

def build():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("transform_query", transform_query)  # transform_query
    
    workflow.add_node("route", route)
    workflow.add_node("web_search", search_web)
    
    workflow.add_node("retrieve", retrieve)  # retrieve    
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate

    workflow.add_node("hallucination_grade", grade_hallucinations)
    workflow.add_node("answer_grade", grade_answer)
    
    workflow.add_node("end", end)

    # Build graph
    workflow.add_edge(START, "transform_query")
    workflow.add_edge("transform_query", "route")
    workflow.add_conditional_edges(
        "route",
        vectorstore_v_web_search,
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )
    workflow.add_edge("web_search", "end")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", "hallucination_grade")
    workflow.add_edge("hallucination_grade", "answer_grade")
    workflow.add_conditional_edges(
        "answer_grade",
        grade_generation_v_documents_and_question,
        {
            "crazy": "web_search",
            "useful": "end",
            "not useful": "web_search",
        },
    )
    workflow.add_edge("end", END)

    # Compile
    app = workflow.compile()

    return app