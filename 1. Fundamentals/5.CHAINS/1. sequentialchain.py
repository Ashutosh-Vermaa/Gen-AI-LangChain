from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Tell 3 fun facts about {topic}",
    input_variable=['topic']
)

parser=StrOutputParser()

chain= template | model | parser

chain.get_graph().print_ascii()  
""" Visualizing chain"""

result=chain.invoke({'topic':'Bangalore'})
print(result)