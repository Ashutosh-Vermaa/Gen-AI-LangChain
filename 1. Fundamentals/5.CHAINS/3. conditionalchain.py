from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

"""
Task- given feedback, do following tasks-
1. anslyse whether it is negative or positive sentiment
2. Based on the sentiment, write a message back to the user.
"""


llm = HuggingFaceEndpoint(
    repo_id="microsoft/phi-4",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]= Field(description= "Return positive, negtaive or neutral based on the sentiment of the customer review")

parser=PydanticOutputParser(pydantic_object=Sentiment)

prompt=PromptTemplate(
    template="Return the sentiment of the review {review} \n {format_instructions}",
    input_variables=['review'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain= prompt | model | parser

pos_prompt=PromptTemplate(
    template="Write an appropriate to this positive feedback \n {feedback}",
    input_variables=['feedback']
)
neg_prompt=PromptTemplate(
    template="Write an appropriate to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

output_parser=StrOutputParser()
branch_chain=RunnableBranch(
    # (condition1, chain to execute),
    # (condition2, chain to execute),
    # default chain
    (lambda x: x.sentiment in ["positive", "neutral"], pos_prompt | model | output_parser),
    (lambda x: x.sentiment=="negative", neg_prompt | model | output_parser),
    RunnableLambda(lambda x: "couldn't find sentiment") # default chain

)

review="""
I would say really this is the best phone and this..... At the price you are getting this this is totally phenomenal because the camera is average but battery is awesome and performance is also awesome.. One thing could have been better the charger is given only 10 Watts in the box.. But still can be charged by another charger.
Performance 10
Camera 8
Display 9
Battery 10
"""

merge_chain=chain | branch_chain

result=merge_chain.invoke({'review':"The phone is good, battery can be better"})

print(result)

merge_chain.get_graph().print_ascii()