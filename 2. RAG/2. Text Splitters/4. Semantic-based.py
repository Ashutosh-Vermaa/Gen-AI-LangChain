from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text="""
Former India bowling coach Bharat Arun believes that if current Test vice-captain Rishabh Pant can adopt some of the traits of his idol MS Dhoni, he has the potential to become an excellent captain.

India are set to face England in a challenging five-match Test series starting 20 June. The series marks the beginning of a new era in Indian cricket, with Shubman Gill taking over as captain and Pant stepping in as his deputy.

While Pant has led in the IPL, he is yet to be part of the leadership group in the national team. As he embarks on this new role, Arun feels that Pant’s naturally composed demeanour could serve him well in a leadership capacity, drawing comparisons with the legendary Dhoni.
"""

splitter=SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation", #std dev of similarities
    breakpoint_threshold_amount=1 #if more than 1 std. dev, start a new chunk
)

result=splitter.create_documents(text)
print(len(result))
print(result)
