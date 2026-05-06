import time
from pprint import pprint
import graph.graph_build as graph_build

def graph(question):
    app = graph_build.build()
    
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])

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