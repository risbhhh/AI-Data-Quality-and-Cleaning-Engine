# LangChain client stub. This file contains a safe, optional helper that uses LangChain.
# The actual LLM call is optional and disabled by default. If you want to enable,
# set OPENAI_API_KEY in your environment and ensure 'openai' is installed.

from langchain import LLMChain, PromptTemplate
try:
    from langchain.llms import OpenAI
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

template = """You are a data-cleaning assistant.
Column: {col}
Stats: {stats}
Propose up to 3 imputation strategies with one-line pros/cons and a short pandas code snippet for each."""

prompt = PromptTemplate(input_variables=['col','stats'], template=template)

def suggest_imputations(col:str, stats:dict):
    if not LLM_AVAILABLE:
        return 'LLM unavailable: install langchain + openai and set OPENAI_API_KEY.'
    llm = OpenAI(temperature=0.2)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(col=col, stats=str(stats))
