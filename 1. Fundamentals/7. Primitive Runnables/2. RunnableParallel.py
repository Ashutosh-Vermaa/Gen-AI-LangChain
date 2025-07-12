from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

"""
Task- given a topic, generte a tweet for X and a linkedin post
"""

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

tweet_prompt=PromptTemplate(
    template="Generate a tweet on {topic}",
    input_variables=['topic'] 
)
linkedin_prompt=PromptTemplate(
    template="Write a linkedin post on {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()
chain=RunnableParallel({
    'tweet': tweet_prompt | model | parser,
    'linkedin_post': linkedin_prompt | model | parser
})

result=chain.invoke({'topic':'Agentic AI'})

print(result)