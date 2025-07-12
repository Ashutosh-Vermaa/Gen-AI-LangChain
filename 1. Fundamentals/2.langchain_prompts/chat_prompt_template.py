from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_template=ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert"),
    ('human',"Explain {topic} in simple terms")
])

prompt=chat_template.invoke({
    'domain':'cricket',
    'topic': 'DRS' 
})

print(prompt)