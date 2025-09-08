import os
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate

def generate_cleaning_script(profile: dict, df_head: str) -> str:
    """Use Hugging Face model to auto-generate a Pandas cleaning script"""
    template = """
    You are a Python data cleaning assistant.
    The dataset has the following issues:
    {issues}

    Here is the dataset head:
    {df_head}

    Generate a Pandas cleaning script that fixes these issues step by step.
    """

    prompt = PromptTemplate(template=template, input_variables=["issues", "df_head"])

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_length": 512},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    response = llm.invoke(prompt.format(issues=profile, df_head=df_head))
    return response
