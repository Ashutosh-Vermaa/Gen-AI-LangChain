# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# import os
# os.environ['HF_HOME']="D:/huggingface_cache"

# llm=HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs={
#         'temperature':0.5,
#         'max_new_tokens':100
#     }
# )

# model=ChatHuggingFace(llm=llm)

# result=model.invoke("what is capital of India?")
# print(result)

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Local model directory (point directly to the snapshot folder)
local_model_path = "D:/huggingface_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

# Load tokenizer and model from local path
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForCausalLM.from_pretrained(local_model_path)

 Create the Hugging Face pipeline#
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.5, max_new_tokens=100)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

# Run a test query
result = chat_model.invoke("What is the capital of India?")
print(result.content)  # Use .content to get the actual output
