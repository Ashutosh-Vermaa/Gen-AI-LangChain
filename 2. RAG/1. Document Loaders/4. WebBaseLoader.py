from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader, TextLoader

loader=WebBaseLoader("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
docs=loader.load()

print(len(docs))

"""
Doing Q&A on this
"""
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt=PromptTemplate(
    template="Answer the following question: {question} from the text :{text}",
    input_variables=['question','text']
)
parser=StrOutputParser()

llm=ChatOpenAI()

chain = prompt | llm | parser

result=chain.invoke({'question':"which is the model here and summary about it", 'text':docs[0].page_content})

print(result)