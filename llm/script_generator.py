import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def generate_cleaning_script(profile: dict, df_head: str) -> str:
    """Use LLM to auto-generate a Pandas cleaning script"""
    template = """
    You are a Python data cleaning assistant.
    The dataset has the following issues:
    {issues}

    Here is the dataset head:
    {df_head}

    Generate a Pandas cleaning script that fixes these issues step by step.
    """

    prompt = PromptTemplate(template=template, input_variables=["issues", "df_head"])
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt.format(issues=profile, df_head=df_head))
    return response.content
