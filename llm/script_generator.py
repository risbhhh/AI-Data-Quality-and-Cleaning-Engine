import os
from transformers import pipeline
from langchain.prompts import PromptTemplate

def generate_cleaning_script(profile: dict, df_head: str) -> str:
    """Use Hugging Face directly to auto-generate a Pandas cleaning script"""

    template = """
    You are a Python data cleaning assistant.
    The dataset has the following issues:
    {issues}

    Here is the dataset head:
    {df_head}

    Generate a Pandas cleaning script that fixes these issues step by step.
    """

    prompt = PromptTemplate(template=template, input_variables=["issues", "df_head"])
    final_prompt = prompt.format(issues=profile, df_head=df_head)

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    result = generator(final_prompt, max_length=512, temperature=0.2)
    return result[0]["generated_text"]
