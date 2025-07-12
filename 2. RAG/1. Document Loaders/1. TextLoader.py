from langchain_community.document_loaders import TextLoader

loader= TextLoader("cricket.txt", encoding="utf-8")

docs=loader.load() # returns a list of Document objects
# docs can also be converted into a runnable and added to the chain
print(docs)
print(len(docs))
print(docs[0].metadata)

# USING THIS WITH LLM

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt=PromptTemplate(
    template="Summarise the following text in 3 lines: \n {text}",
    input_variables=['text']
)
parser=StrOutputParser()

llm=ChatOpenAI()

chain= prompt | llm | parser

result=chain.invoke({'text':docs[0].page_content})
print(result)