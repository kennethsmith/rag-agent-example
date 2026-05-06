from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

def retrieval_grader():
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    return prompt
    
def get_retrieval_grader(llm):
    rg = retrieval_grader() | llm | StrOutputParser()
    return rg

def get_retrieval_grader_json(llm):
    rg = retrieval_grader() | llm | JsonOutputParser()
    return rg

def invoke_retrieval_grader(grader, question, document):
    grade = grader.invoke({"question": question, "document": document})
    return grade


def hallucination_grader():
    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "documents"],
    )
    return prompt

def get_hallucination_grader(llm):
    hg = hallucination_grader() | llm | StrOutputParser()
    return hg

def get_hallucination_grader_json(llm):
    hg = hallucination_grader() | llm | JsonOutputParser()
    return hg

def invoke_hallucination_grader(grader, docs, generation):
    ### Hallucination Grader
    grade = grader.invoke({"documents": docs, "generation": generation})
    return grade


def answer_grader():
    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["generation", "question"],
    )
    return prompt

def get_answer_grader(llm):
    ag = answer_grader() | llm | StrOutputParser()
    return ag

def get_answer_grader_json(llm):
    ag = answer_grader() | llm | JsonOutputParser()
    return ag

def invoke_answer_grader(grader, question, generation):
    ### Answer Grader
    grade = grader.invoke({"question": question, "generation": generation})
    return grade