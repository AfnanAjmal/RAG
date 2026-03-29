# Similarity Search - converts your query into a vector using the embedding model and then compares it against all stored vectors in the database using cosine similarity (or l2/dot product), 
# returns the top k most similar chunks based on the angle between vectors. it is fast, simple and works well for straightforward and specific questions where the wording of your query 
# closely matches the wording in your documents. it is the default search type in chroma and most vector databases.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


doc = [
    Document(page_content="Deep learning is a subset of machine learning that focuses on neural networks with many layers.", metadata={"source": "Document 1"}),
    Document(page_content="Machine learning is a broader field that encompasses various algorithms and techniques for data analysis and prediction.", metadata={"source": "Document 2"}),
    Document(page_content="Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.", metadata={"source": "Document 3"}),
    Document(page_content="Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data.", metadata={"source": "Document 4"}),

]


embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})

vectorstore = Chroma.from_documents(doc, embedding)


similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


result_similarity = similarity_retriever.invoke("What is deep learning?")
print("Similarity Retriever Results:")
for res in result_similarity:
    print(res.page_content)
    print(res.metadata)
    print()
    print()
    print()


