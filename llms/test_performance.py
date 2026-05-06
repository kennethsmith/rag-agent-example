import time
import get as llms_get

def main():    
    ls_start_time = time.perf_counter()
    for i in range(1):
        llm = llms_get.lm_studio()
        response = llm.invoke("Wake up!")
        print(response.content)
    ls_end_time = time.perf_counter()
    print(f"LM Studio lapsed time: {ls_end_time - ls_start_time:.4f} seconds")

    o_start_time = time.perf_counter()
    for i in range(1):
        llm = llms_get.ollama()
        response = llm.invoke("Wake up!")
        print(response.content)
    o_end_time = time.perf_counter()
    print(f"Ollama elapsed time: {o_end_time - o_start_time:.4f} seconds")


if __name__ == "__main__":
    main()