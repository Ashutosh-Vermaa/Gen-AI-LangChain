from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

loader=CSVLoader(r"D:\Documents\LangChain\2. RAG\Data\Fortune 500 Companies US.csv", encoding="ISO-8859-1")

docs=loader.load()
"""
Each row as a single document
"""
print(len(docs))
print(docs[99])