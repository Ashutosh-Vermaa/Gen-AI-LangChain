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

#schema
json_schema={
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ProductReviewSummary",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of key themes mentioned in the review."
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Sentiment of the review"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review."
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Positive aspects of the product mentioned in the review."
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Negative aspects or criticisms of the product mentioned in the review."
    }
  },
  "required": ["key_themes", "summary"],
  "additionalProperties": False
}

structured_model=model.with_structured_output(json_schema)

result=structured_model.invoke("NORYL GTX LMX310 resin is a low CO2 footprint, conductive, non-reinforced alloy of Polyphenylene Ether (PPE) + Polybutylene Terephthalate (PBT). This injection moldable grade is optimized for primer-less electrostatic painting. NORYL GTX LMX310 resin exhibits high heat performance, low moisture uptake and low warpage. This material is a suitable material for automotive applications such as body panels, service flaps, fenders, trunk lid, and exterior trim.")
print(result)
