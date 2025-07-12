from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()
"""
Task- generate a joke and print how many words it has.
"""
model=ChatOpenAI()

prompt=PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=['topic']
)
parser=StrOutputParser()

def word_counter(text):
    return len(text.split())

joke_chain= RunnableSequence(prompt, model, parser)
parallel_chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    # "word_count": RunnableLambda(word_counter)
    "word_count":RunnableLambda(lambda x: len(x.split())) #alternative way
})

chain= RunnableSequence(joke_chain, parallel_chain)

result=chain.invoke({"topic":"politics"})

print(result)