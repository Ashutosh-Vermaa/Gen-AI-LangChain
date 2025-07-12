from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Load PDF files
pdf_loader = DirectoryLoader(
    path=r"D:\Documents\LangChain\2. RAG\Data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

# Load TXT files
txt_loader = DirectoryLoader(
    path=r"D:\Documents\LangChain\2. RAG\Data",
    glob="**/*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")

)
# Load all documents
docs = pdf_loader.load() + txt_loader.load()

print(len(docs))
print(docs[0].page_content)

"""
LAZY LOADING
"""
from itertools import chain
docs = chain(pdf_loader.lazy_load(), txt_loader.lazy_load())
for doc in docs:
    print(doc.metadata)