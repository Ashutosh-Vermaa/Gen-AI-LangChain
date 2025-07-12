from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
# dimension decides the length of the output vector

result=embedding.embed_query("Delhi is capital of India")

print(str(result))