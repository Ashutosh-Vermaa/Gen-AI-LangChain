from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

"""
Task- generate a report on a topic and give that to the llm to generate summary
"""

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B",
    task="text-generation"
)

# model=ChatHuggingFace(llm=llm)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
# prompt
template= PromptTemplate(
    template="Give me the name, age, city of a finctional person. \n {format_instruction} ",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()})
"""
it's called partial_variable because it is not provided by user, but filled before hand.
get_format_instructions() of the parser tells the format of the output
"""
prompt=template.format()
print(prompt)

result=model.invoke(prompt)

print(result)

final=parser.parse(result.content)
print(final)

# alternative way to do it using chain
chain= template | model | parser
result= chain.invoke({})

print(result)

# prompt 2
template2= PromptTemplate(
    template= "Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()})

# prompt2=template2.invoke({'topic':'cricket'})
chain= template2 | model | parser
result= chain.invoke({'topic':'cricket'})

print(result)
# res=model.invoke(prompt2)
# print(res)
# print(parser.parse(res.content))