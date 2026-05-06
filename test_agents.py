import time
from pprint import pprint
import artifacts.get as get
import llms.get as llms_get
import vector.store as vs
import agents.standard as standard
import agents.grader as grader
import agents.rewriter as rewriter
import agents.router as router
import agents.web_search as web_search

def linear(question):
    # Load the posts
    posts = get.posts()
    
    # Load the LLM
    llm = llms_get.llm()
    
    # Rewrite the question in a better way
    q = rewriter.get(llm)
    new_question = rewriter.invoke(q, question)
    print("NEW QUESTION:") 
    pprint(new_question)
    question = new_question
    
    # Test the router response
    r = router.get(llm)
    route = router.invoke(r, question)
    print("ROUTE:")
    pprint(route)
    
    ws = web_search.web_search({"question": question})
    print("WEB SEARCH:")
    pprint(ws)
    
    # Build the RAG VectorStore and get the retriever
    retriever = vs.index(posts)
    # Get the relevant doc for the question
    docs = vs.get(question, retriever)
    
    # Grade the doc content retrieved
    g = grader.get_retrieval_grader(llm)
    for doc in docs:
        r_grade = grader.invoke_retrieval_grader(g, question, doc.page_content)
        print('DOC:')
        pprint(doc.page_content)
        print("r_grade: ", r_grade)
    
    # Create the RAG Chain generation
    generate = standard.get(llm)
    generation = standard.invoke(generate, question, docs)
    print("GENERATION (RAG-CHAIN):")
    pprint(generation)
    
    # Grade the generations for hallucinations
    g = grader.get_hallucination_grader(llm)
    h_grade = grader.invoke_hallucination_grader(g, docs, generation)
    print("h_grade: ", h_grade)
    
    # Grade the generation on the answer
    g = grader.get_answer_grader(llm)
    a_grade = grader.invoke_answer_grader(g, question, generation)
    print("a_grade: ", a_grade)
    

def main():
    # Seed the question
    question = "agent memory"
    # question = "Explain how the different types of agent memory work?"
    
    linear_start_time = time.perf_counter()
    linear(question)
    linear_end_time = time.perf_counter()

    print(f"Linear elapsed time: {linear_end_time - linear_start_time:.4f} seconds")

if __name__ == "__main__":
    main()