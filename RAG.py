import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"})

vectorstore = Chroma(
    persist_directory="chroma-db", 
    embedding_function=embedding_model
    )

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """You are a helpful AI assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context,
        say: I could not find the answer in the document"""),
        ("human","""
        Question: {question}
        Context: {context}
        """)
    ]
)

print("RAG System Ready!")
print("Press 0 to exit.")

while True:
    query = input("You: ")
    if query == "0":
        print("Exiting...")
        break
    
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
        )
    
    final_prompt = prompt_template.invoke(
        {"context": context, "question": query}
        )

    response = llm.invoke(final_prompt)
    
    print(f"\nAI: {response.content}")  
    print("\n" + "="*50 + "\n")