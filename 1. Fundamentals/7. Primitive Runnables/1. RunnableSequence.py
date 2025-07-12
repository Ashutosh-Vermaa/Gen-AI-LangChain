from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

"""
Task- generate a joke and ask to rate it
"""

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Write a one liner joke about {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()

prompt2=PromptTemplate(
    template="Provide the {joke} and then rate it- funny, unfunny",
    inuput_variables=['joke']
)

chain= RunnableSequence(prompt1, model,parser, prompt2, model, parser)
result=chain.invoke({'topic':"politics"})
print(result)