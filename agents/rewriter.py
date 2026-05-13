import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get(llm):
    # Prompt
    re_write_prompt = PromptTemplate(
        template="""You are a question re-writer that converts an input question to a 
            better version that is optimized for vectorstore retrieval. Look at the 
            initial and formulate an improved question.
            
            The response should be in valid JSON format in string form with a single key of 'new_question'.
            Do not add any additional context to the response.

            Here is the initial question: {question}.  \n """,
        input_variables=["question"],
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

def invoke(rewriter, question) -> str:
    new_question = rewriter.invoke({"question": question})
    return json.loads(new_question)["new_question"]