from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text="""
class Greeter:
    def greet(self, name):
        print(f"Hello, {name}!")

# Create an object of the class
my_greeter = Greeter()

# Call the method
my_greeter.greet("Ashutosh")

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0, #takes mentioned number of last characters from the previous chunk as the starting point
)

result=splitter.split_text(text)
print(result)

"""
Similarly can be done for Markdown
"""
