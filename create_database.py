#load pdf
#create chunks
#creating the embeddings
#store the chunks in chroma db

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()

#load the pdf document
doc = PyPDFLoader("Document_loader/DeepLearning.pdf")
data = doc.load()

#split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
    )
chunk = splitter.split_documents(data)

#creating the embeddings
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", 
                                        model_kwargs={"device": "cpu"} 
                                        )

#store the chunks in chroma db
VectorStore = Chroma.from_documents(
    documents=chunk,
    embedding=embedding_model,
    persist_directory="chroma-db",
)   

print("Vector store created successfully!")