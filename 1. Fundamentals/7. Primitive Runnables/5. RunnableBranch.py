from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch

load_dotenv()
"""
Task- Generate a report on a topic, summarise if >500 words else return as it is.
"""
model=ChatOpenAI()

report_gen_prompt=PromptTemplate(
    template="Write a report on {topic}",
    input_variable=["topic"]
)
parser=StrOutputParser()

report_gen_chain=RunnableSequence(report_gen_prompt, model, parser)

summarise_prompt=PromptTemplate(
    template="Summarize the following text: \n {text}",
    input_variables=['text']
)

conditional_chain=RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(summarise_prompt, model, parser)),
    #(condition, chain call) #format of the input
    RunnablePassthrough()
)

chain=RunnableSequence(report_gen_chain, conditional_chain)
result=chain.invoke({'topic':"Sanchi Stupa"})

print(result)