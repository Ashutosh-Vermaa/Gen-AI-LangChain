from langchain.text_splitter import CharacterTextSplitter

text="""
Former India bowling coach Bharat Arun believes that if current Test vice-captain Rishabh Pant can adopt some of the traits of his idol MS Dhoni, he has the potential to become an excellent captain.

India are set to face England in a challenging five-match Test series starting 20 June. The series marks the beginning of a new era in Indian cricket, with Shubman Gill taking over as captain and Pant stepping in as his deputy.

While Pant has led in the IPL, he is yet to be part of the leadership group in the national team. As he embarks on this new role, Arun feels that Pant’s naturally composed demeanour could serve him well in a leadership capacity, drawing comparisons with the legendary Dhoni.
"""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0, #takes mentioned number of last characters from the previous chunk as the starting point
    separator=""
)

result=splitter.split_text(text)
# print(result)

"""
Combining document loader and text splitter
"""
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"D:\Documents\LangChain\2. RAG\Data\DOC-20240917-WA0005.pdf")
docs=loader.load()

print(len(docs))

result=splitter.split_documents(docs)

print(result[0]) #each chunk will be a Document object 