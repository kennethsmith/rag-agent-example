### Router

from pprint import pprint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def get(llm):
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. \n
        Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
        You do not need to be stringent with the keywords in the question related to these topics. \n
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
        Return the a JSON with a single key 'datasource' and no preamable or explanation. \n
        Question to route: {question}""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    return question_router

def invoke(router, question):
    route = router.invoke({"question": question})
    return route