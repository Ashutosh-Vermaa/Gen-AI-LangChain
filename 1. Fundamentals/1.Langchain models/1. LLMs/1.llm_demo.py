from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

result=llm.invoke("how old is he?")

print(result)

# # GPT4
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# response = llm([HumanMessage(content="what is the captial of India?")])
# print(response.content)