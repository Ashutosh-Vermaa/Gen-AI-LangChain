from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Write a one liner joke about {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()
chain1= prompt1 | model | parser
result= chain1.invoke({'topic': "friendship"})

prompt2=PromptTemplate(
    template="How did you find this joke- funny, unfunny \n {joke}",
    inuput_variables=['joke']
)

chain2=RunnableParallel({
    "joke": RunnablePassthrough(),
    "Rating": prompt2 | model | parser
})

merge_chain=chain1 | chain2
result=merge_chain.invoke({'topic':"friendship"})
print(result)