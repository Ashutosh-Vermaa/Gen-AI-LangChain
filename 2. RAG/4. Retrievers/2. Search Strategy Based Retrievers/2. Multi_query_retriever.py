from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Create LangChain documents for IPL players

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
#initialize embedding
embedding_model=OpenAIEmbeddings()

# create CHroma vector store in memory
vectorstore=FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

#convert vector store into a retriever
similarity_retriever=vectorstore.as_retriever(
    search_type="similarity", #standard similarity search
    search_kwargs={'k':5}) 

#MQR retriever
multiquery_retriever= MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k':5}),
    llm=ChatOpenAI()
)
#Query the database using similarity_retriever
query="How to improve energy levels and maintain balance?"
similarity_results=similarity_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n####Results {i+1}######\n")
    print(doc.page_content)

#Query the database using similarity_retriever
query="How to improve energy levels and maintain balance?"
multiquery_results=multiquery_retriever.invoke(query)

for i, doc in enumerate(multiquery_results):
    print(f"\n####Results {i+1}######\n")
    print(doc.page_content)