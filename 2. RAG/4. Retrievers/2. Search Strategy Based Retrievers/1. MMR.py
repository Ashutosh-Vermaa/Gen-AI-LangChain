from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Create LangChain documents for IPL players

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="LangChain is used to build LLM based real wordl applications."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
#initialize embedding
embedding_model=OpenAIEmbeddings()

# create CHroma vector store in memory
vectorstore=FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

#convert vector store into a retriever
retriever=vectorstore.as_retriever(
    search_type="mmr", #this enables MMR
    search_kwargs={'k':3, "lambda_mult":0.5}) #lambda_mult for MMR is like temperature in LLM, 1 means works as normal similarity search, 0 provides very diverse results

#Query the database
query="what is Langchain"
results=retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n####Results {i+1}######\n")
    print(doc.page_content)