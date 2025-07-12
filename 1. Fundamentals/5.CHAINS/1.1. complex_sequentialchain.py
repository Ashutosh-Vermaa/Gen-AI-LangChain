from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

report_prompt=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variable= ['topic']
)
summary_prompt=PromptTemplate(
    template="Generate a 5 point summary of the following text: \n {text}",
    input_variable=['text']
)

parser=StrOutputParser()

chain= report_prompt | model | parser | summary_prompt | model | parser

# result=chain.invoke("Narsinghgarh Fort")

# print(result)

# chain.get_graph().print_ascii()

print(model.invoke("are you aware of Narsinghgarh Fort situated in Madhya Pradesh? Provide details about it"))