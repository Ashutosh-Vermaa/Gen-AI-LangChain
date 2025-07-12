from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field

load_dotenv()
model=ChatOpenAI()

#schema
class Review(BaseModel):
    summary: str #Pydantic enforces validation, summary can only be string type now
    sentiment: str

# Annotated 
class AnnotatedReview(BaseModel):

    features: list[str]= Field(description= "A list of all the features of the product")
    Applications: list[str]= Field(description= "A list of all the applications of the product")
    sentiment: str= Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]]= Field(default=None, description= "Provide the pros of the product from the review")
    cons: Optional[list[str]]= Field(default=None, description= "Provide the cons of the product from the review")

    # # key_themes: Annotated[list[str], "provide key themes discussed in the review in a list"]
    # features: Annotated[list[str], "A list of all the features of the product"]
    # Applications: Annotated[list[str], "A list of all the applications of the product"]
    # # summary: Annotated[str, "A brief overview of the main points"] 
    # # we can provide brief description of the expected output for the model to understand
    # sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]
    # pros: Annotated[Optional[list[str]], "Provide the pros of the product from the review"]
    # cons: Annotated[Optional[list[str]], "Provide the cons of the product from the review"]

structured_model=model.with_structured_output(AnnotatedReview)

result=structured_model.invoke("NORYL GTX LMX310 resin is a low CO2 footprint, conductive, non-reinforced alloy of Polyphenylene Ether (PPE) + Polybutylene Terephthalate (PBT). This injection moldable grade is optimized for primer-less electrostatic painting. NORYL GTX LMX310 resin exhibits high heat performance, low moisture uptake and low warpage. This material is a suitable material for automotive applications such as body panels, service flaps, fenders, trunk lid, and exterior trim.")
print(result.features)
