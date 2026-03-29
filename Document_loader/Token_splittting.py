from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv


data = PyPDFLoader("Document_loader/Data-Science.pdf").load()

splliter = TokenTextSplitter(chunk_size=10, chunk_overlap=1)
chunk = splliter.split_documents(data)
print(len(chunk))
for i in chunk:
    print(i.page_content)
    print()
    print()
    print()
    print()