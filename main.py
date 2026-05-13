import time
from pprint import pprint
from llms import get as llm_get
from artifacts import get as a
import vector.store as vs
import graph.graph_build as graph_build

def graph(question):
    llm = llm_get.llm()
    docs = a.posts()
    app = graph_build.build()
    retriever = vs.index(docs)
    
    inputs = {"question": question, "llm": llm, "retriever": retriever}
    outputs = {}
    for output in app.stream(inputs):
        for key, value in output.items():
            outputs[key] = value
    pprint(outputs.keys())
    pprint("Here is what I generated:")
    pprint(outputs['end']['generation'])
    print("\n")
    print("Should you trust it? ", outputs['end']['hallucination_grade']['grade'])
    print("Is it a good answer? ", outputs['end']['answer_grade']['grade'])

def main():
    # Seed the question
    question = "agent memory"
    # question = "Explain how the different types of agent memory work?"
    
    graph_start_time = time.perf_counter()
    graph(question)
    graph_end_time = time.perf_counter()

    print(f"Graph lapsed time: {graph_end_time - graph_start_time:.4f} seconds")

if __name__ == "__main__":
    main()