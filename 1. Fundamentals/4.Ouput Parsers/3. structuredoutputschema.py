from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

schema= [
    ResponseSchema(name='fact1', description=' fact 1 about the topic'),
    ResponseSchema(name='fact2', description=' fact 2 about the topic'),
    ResponseSchema(name='fact3', description=' fact 3 about the topic'),
]
parser=StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}

)

#through chain
chain= template | model | parser
result=chain.invoke({'topic':'cricket sport'})

# manual way
prompt=template.invoke({'topic':'cricket sport'})
result=model.invoke(prompt)
print(result)
final_res=parser.parse(result.content)
print(final_res)