from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import re
from langchain_core.documents import Document
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
import logging
import time

load_dotenv()


# step 1- loading document ----------------------
file_path = r"D:\Documents\LangChain\4. End to End Project\Transcript of 39th AGM FY24.pdf"
loader = PyPDFLoader(file_path)

docs=loader.load()
len(docs)

#pre-processing
pattern = r"Amara Raja Energy & Mobility Limited\s+\(Formerly known as Amara Raja Batteries Limited\)\s+.*, \d+\s+Page \d+ of \d+\s*"
complete_pdf=Document(page_content="")
for doc in docs:
    doc.page_content= re.sub(pattern, "", doc.page_content)
    # doc.page_content=text.replace("\n", "")
    complete_pdf.page_content+= " " + doc.page_content

#-------------------STEP 2- Splitting DOCS------------------------
chunks=re.split(r"(?=\n[A-Z][a-z]+(?: [A-Z][a-z]+)*:\s*)", complete_pdf.page_content)
#splitting at <name>:

final_docs=[]
for doc in chunks:
    doc=re.sub(r"\s+", " ", doc)
    index_colon=doc.find(":") if doc.find(":")!=-1 else -1
    final_docs.append(Document(page_content=doc[index_colon+1:], metadata={'speaker_name':doc[:index_colon]}))

print(final_docs[:4])

#---------------------3. Indexing Documents -----------------------
embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=700)

from pinecone import ServerlessSpec
pc = Pinecone()  # Or Pinecone(api_key=your_key) if not set in env

index_name = "earning-call-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=700,  # <--- Check this carefully!
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embedding)
vector_store.add_documents(final_docs)

#----------------------4. Creating a Retriver ----------------------------
retriever=vector_store.as_retriever(
    search_type="mmr", #this enables MMR
    search_kwargs={'k':5, "lambda_mult":0.5} #lambda_mult for MMR is like temperature in LLM, 1 means works as normal similarity search, 0 provides very diverse results
)

print(retriever.invoke("How much did the company grow compared to last quarter"))

#---------------------5. Categorizing Question- topic vs document level--------------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
import json

class TopicOrDocumentLevel(BaseModel):
    question_level: Literal["document", "topic"] = Field(
        description=(
            "Classifies the user's question as either 'document' or 'topic' level. "
            "'Document' level questions require understanding the entire document to answer — for example, questions about the overall summary, key takeaways, or main themes. "
            "'Topic' level questions focus on a specific detail or section and can typically be answered using only part of the document."
        )
    )

schema=PydanticOutputParser(pydantic_object=TopicOrDocumentLevel)

# creating prompt
question_classifier_prompt = PromptTemplate(
    template="""
    You are a knowledgeable financial assistant with expertise in analyzing company earnings calls. 
    Your task is to classify the following user question into one of two categories: "document-level" or "topic-level".

    - A **document-level** question requires understanding or summarizing the entire earnings call transcript. 
      Examples include questions about the overall summary, key takeaways, or general themes discussed in the call.

    - A **topic-level** question focuses on a specific section, detail, or aspect of the call and can usually be answered by referencing only a portion of the transcript.

    Use the following format when providing your classification:
    {format_instructions}

    Question: {question}
    """,
    input_variables=["question"],
    partial_variables={'format_instructions': schema.get_format_instructions()}
)


llm=ChatOpenAI()
parser=StrOutputParser()

question_classifier_chain= question_classifier_prompt | llm | parser
result=question_classifier_chain.invoke({"question": "what are the future prospects of growth"})
print("classified question:", result)

############ NEXT CREATING CONDITIONAL CHAIN BASED ON QUESTIONS CATEGORY
from langchain.prompts import PromptTemplate

response_prompt = PromptTemplate(
    template="""
You are a knowledgeable financial assistant specialized in interpreting company earnings calls.
Use only the information provided in the context below to answer the user's question.

If the answer cannot be found in the context, respond with:
"Sorry, I don't have enough context to answer this question."

Context:
{context}

Question:
{question}

Answer in a concise and accurate manner, quoting from the context where relevant.
If the context does not contain enough information, do not attempt to answer or guess. Do not use prior knowledge or assumptions.
""",
    input_variables=['context', 'question']
)


# case 1: topic level question
def preprocessing(docs):
    return "\n".join([doc.page_content for doc in docs])

topic_chain= retriever  \
                        |RunnableParallel( 
                            {'context': RunnableLambda(preprocessing),
                             'question': RunnablePassthrough()
                             }) 
                        

# case 2: document level question
"""
1. Need to pass the complete document- summarise chunks and concat them
"""

summary_prompt=PromptTemplate(
    template="""
    You are a knowledgeable financial assistant specialized in interpreting company earnings calls.
    Summarise the below text to give important details.

    text: {text}
    """, input_variables=['text']
)
summary_chain=summary_prompt | llm

summarised_docs=[summary_chain.invoke({'text': doc.page_content}) for doc in final_docs]
single_doc="\n".join([doc.content.replace("Summary:", "").replace("\n", "") for doc in summarised_docs]) 

######## COmbining two CASES

def is_topic_level(question: str) -> bool:
    result = question_classifier_chain.invoke({'question': question})
    parsed = json.loads(result)
    return parsed.get("question_level") == "topic"

common_chain= response_prompt | llm | parser

final_chain=RunnableBranch(
    (is_topic_level, topic_chain | common_chain),
    (lambda q : {'question': q, 'context': single_doc}) |common_chain #default chain
)

question="key takeways from the report"
print(final_chain.invoke(question))

