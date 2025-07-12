from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=2, lang='en')

query="History of India Vs Pakistan"
docs=retriever.invoke(query)

print(len(docs))
for doc in docs:

    print(doc.page_content, "\n\n\n")
