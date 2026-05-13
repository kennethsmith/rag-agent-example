from IPython.display import Image
import graph.graph_build as graph_build

def main():
    app = graph_build.build()
    print(app.get_graph().draw_ascii())
    diagram = Image(app.get_graph().draw_mermaid_png())
    with open("diagram.png", "wb") as f:
        f.write(diagram.data)

if __name__ == "__main__":
    main()