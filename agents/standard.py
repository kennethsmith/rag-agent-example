from artifacts.get import rlm_rag_prompt as get_prompt
from langchain_core.output_parsers import StrOutputParser

def get(llm):       
    prompt = get_prompt()
    standard = prompt | llm | StrOutputParser()
    return standard

def invoke(generate, question, docs):
    return generate.invoke({"question": question, "context": docs})