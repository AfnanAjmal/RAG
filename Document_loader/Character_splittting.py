from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=1
    )

doc = PyPDFLoader("Document_loader/Data-Science.pdf")
data = doc.load()

chunk = splitter.split_documents(data)

print(len(chunk))
for i in chunk:
    print(i.page_content)
    print()
    print()
    print()
    print()