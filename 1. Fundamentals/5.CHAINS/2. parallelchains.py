from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import RunnableParallel # to run parallel chains

load_dotenv()

"""
Task- given a text, generate notes and quiz from it and merge these two in a single document.

There are two parts to it-
1. Create a parallel chain to generate notes and quiz
2. A sequential (to above task) chain to merge these two
"""

llm1=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)
llm2=HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    task="text-generation"
)
model1=ChatHuggingFace(llm=llm1)
model2=ChatHuggingFace(llm=llm2)

### NOTES
notes_prompt= PromptTemplate(
    template= "Generate Notes from the given text to revise during exams {text}",
    input_variables=['text']
)
quiz_prompt=PromptTemplate(
    template="Generate 5 short question and answers from the text {text}",
    input_variables=['text']
)

merge_prompt=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes-> {notes} \n quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser= StrOutputParser()

parallel_chain=RunnableParallel({
    "notes": notes_prompt |model1 | parser,
    'quiz': quiz_prompt | model2 | parser

})

merge_chain=merge_prompt | model1 | parser
chain= parallel_chain | merge_chain

text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.


"""
result=chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()