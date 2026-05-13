import get as get_llms
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def prompt():
    return PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question", "documents"],
    )
    
def prompt_args():
    return {
        "documents": {
            "1": "Cabins are yellow.",
            "2": "Cabins are painted with yellow paint.",
            "3": "Witness: 'I have only seen yellow cabins.'"
        },
        "generation": "Cabins are yellow."
    }

def test_str_parser():
    grader = prompt() | get_llms.llm() | StrOutputParser()
    grade = grader.invoke(prompt_args())
    print(grade)
    print(grade[0])

def main():
    test_str_parser()

if __name__ == "__main__":
    main()