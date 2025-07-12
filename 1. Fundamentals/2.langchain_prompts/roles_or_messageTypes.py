from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

messages=[SystemMessage("You are a helpful assistant"),
          HumanMessage("Tell me about langchain")]

messages.append(AIMessage(model.invoke(messages).content))

messages.append(HumanMessage("its LangChain"))
messages.append(AIMessage(model.invoke(messages).content))
print(messages)