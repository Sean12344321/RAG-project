import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

url = "https://www.gutenberg.org/cache/epub/24022/pg24022.txt"

response = requests.get(url)
response.raise_for_status()  # Check that the request was successful
text = response.text

documents = [Document(page_content=text)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(documents)