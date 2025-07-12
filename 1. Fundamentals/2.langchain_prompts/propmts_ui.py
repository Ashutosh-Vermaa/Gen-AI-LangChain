from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatOpenAI(model="gpt-4")

st.header("Research Assistant")
st.write('Provide the inputs')

paper_choice = st.selectbox(
    'Select your model',
     ('BERT', 'Word2Vec', 'Attention is all you need'))

style_choice = st.selectbox(
    'Select your explanation style',
     ('Beginner-friendly', 'Technical', 'Code-oriented', "Mathematical"))

length_choice = st.selectbox(
    'Select your explanation length',
     ('short (1-2 paragraph)', 'Medium (3-5 paragraph)', "Long (detailed-explanation)"))


template=PromptTemplate(
    template="""
    Please summarize the research paper titled {paper_choice} with following instructions:
    Explanation style: {style_choice}
    Explanation Length: {length_choice}

    1. Mathematical detail:
        - include relevant mathematical equations if present in the paper
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
        - use relatable analogies to simplify complex tasks
    If certain iformation is not available, respond with "Insufficient information available" instead of guessing
    Ensure the summary is clear, accurate and aligned with provided style and style.  
    """,
    input_variables=['paper_choice', 'style_choice', 'length_choice']
    )

prompt=template.invoke({
    'paper_choice': paper_choice,
    'style_choice': style_choice,
    'length_choice':length_choice
}
)

if st.button("Summarize"):
    result=model.invoke(prompt)
    st.write(result.content)
