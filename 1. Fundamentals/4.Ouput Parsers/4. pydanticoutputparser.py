from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

#schema
class Person(BaseModel):
    name: str= Field(description= "Name of the Person")
    age : int = Field(gt=18, description= "Age of the person")
    city: str = Field(description= "Name of the city")

parser= PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Generate name, age, city of a fictional {place} Person \n {format_instructions}",
    input_variables=['place'],
    partial_variables= {'format_instructions': parser.get_format_instructions()} 
)

prompt=template.invoke({"place": "Indian"})

print(prompt) 
"""to see what we are sending to LLM behind the scene"""
# res=model.invoke(prompt)
# final_res=parser.parse(res.content)
# print(final_res)

# using chain
chain= template | model | parser
res=chain.invoke({'place':"Indian"})
print(res)
