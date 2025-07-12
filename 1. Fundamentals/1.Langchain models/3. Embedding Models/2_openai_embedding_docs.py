from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
# dimension decides the length of the output vector

documents=["bangalore is a city in India",
            "rain water harvesting is a fantastic practice ",
            "red is a color of sacrifice"
            ]

result=embedding.embed_documents(documents)

print(str(result))