from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader(r"D:\Documents\LangChain\2. RAG\Data\DOC-20240917-WA0005.pdf")

docs=loader.load() 
print(docs)

"""

"""