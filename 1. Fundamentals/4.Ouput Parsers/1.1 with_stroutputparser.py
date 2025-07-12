
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
Most common usage of StrOutputParser is with chains
"""

load_dotenv()

"""
Task- generate a report on a topic and give that to the llm to generate summary
"""

# model=ChatHuggingFace(llm=llm)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

# prompt 1
tempate1= PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variable=['topic']
)

#prompt 2
tempate2= PromptTemplate(
    template="Write a 5 line summary of the following text. /n {text}",
    input_variable=['text']
)


parser= StrOutputParser()

chain= tempate1 | model | parser | tempate2 | model | parser

result= chain.invoke({'topic': 'black hole'})

print(result)