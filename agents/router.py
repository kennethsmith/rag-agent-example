import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get(llm):
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search. 
            Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. 
            You do not need to be stringent with the keywords in the question related to these topics. 
            Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
            
            The response should be in valid JSON format in string form with a single key of 'datasource'.
            Do not add any additional context to the response.
            
            Question to route: {question}""",
        input_variables=["question"],
    )

    question_router = prompt | llm | StrOutputParser()
    return question_router

def invoke(router, question) -> dict[str]:
    route = router.invoke({"question": question})
    return json.loads(route)