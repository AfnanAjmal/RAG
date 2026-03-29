# MMR (Max Marginal Relevance) - works exactly like similarity search under the hood but adds an extra step after finding similar chunks. instead of just returning the top k most similar results,
# it checks if the results are too similar to each other and penalizes redundant chunks, so the final results are both relevant to your query AND diverse from each other. 
# best used when your documents have a lot of repeated or overlapping information and you want varied results that cover different aspects of your query without repetition.

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


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
    )

result_mmr = mmr_retriever.invoke("What is deep learning?")
print("MMR Retriever Results:")
for res in result_mmr:
    print(res.page_content)
    print(res.metadata)
    print()
    print() 
    print()

