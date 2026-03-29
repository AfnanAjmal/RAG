# Multi Query - instead of searching the vector db with just your single query, it first passes your query to an LLM which generates multiple different versions and phrasings of your question.
# for example "what is deep learning" becomes 3-5 different queries like "how do neural networks learn", "explain multilayer network training", "fundamentals of DL" etc.
# all these queries are then used to search the vector db separately and the results are merged together with duplicates removed. this gives much better coverage and recall because
# a single query might miss chunks that are worded differently, but one of the generated queries will likely catch them. it is the slowest of the three since it requires an LLM call
# but gives the best results for complex, broad or ambiguous questions where exact wording of the query may not match the wording in your documents.
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

docs = [
    Document(page_content="Deep learning is a subset of machine learning that focuses on neural networks with many layers.", metadata={"source": "Document 1"}),
    Document(page_content="Machine learning is a broader field that encompasses various algorithms and techniques for data analysis and prediction.", metadata={"source": "Document 2"}),
    Document(page_content="Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.", metadata={"source": "Document 3"}),
    Document(page_content="Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data.", metadata={"source": "Document 4"}),
]


embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})


vectorstore = Chroma.from_documents(docs, embedding)

retriever = vectorstore.as_retriever()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

Multi_query_retriever = MultiQueryRetriever.from_llm(  # ✅ called on class not instance
    llm=llm,
    retriever=retriever
)

query = "What is deep learning"
result_multi_query = Multi_query_retriever.invoke(query)
print("Multi Query Retriever Results:")
for res in result_multi_query:
    print(res.page_content)
    print(res.metadata)
    print()
    print()
    print()
