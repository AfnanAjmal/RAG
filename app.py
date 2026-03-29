import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

* { font-family: 'DM Mono', monospace; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background-color: #0e0e0e;
    color: #e8e8e8;
}

section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

.stChatMessage {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    margin-bottom: 12px !important;
}

.stChatInput textarea {
    background-color: #1a1a1a !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background-color: #d4f54a !important;
    color: #0e0e0e !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
    width: 100% !important;
}

.stButton > button:hover {
    background-color: #c2e03a !important;
}

.stFileUploader {
    background-color: #1a1a1a !important;
    border: 1px dashed #3a3a3a !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.stProgress > div > div {
    background-color: #d4f54a !important;
}

.status-box {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #d4f54a;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
    color: #aaa;
}

.title-text {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #e8e8e8;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.accent { color: #d4f54a; }

.subtitle {
    font-family: 'DM Mono', monospace;
    color: #666;
    font-size: 0.85rem;
    margin-top: 4px;
}

.chunk-info {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #888;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

def process_pdf(uploaded_file):
    """Save uploaded PDF, chunk it, embed it, store in Chroma."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(data)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db",
    )

    os.unlink(tmp_path)
    return vectorstore, len(chunks)

def load_existing_vectorstore():
    """Load already-embedded chroma-db if it exists."""
    if os.path.exists("chroma-db"):
        embedding_model = get_embedding_model()
        return Chroma(
            persist_directory="chroma-db",
            embedding_function=embedding_model
        )
    return None

def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.5}
    )

def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.
     Use ONLY the provided context to answer the question.
     If the answer is not present in the context,
     say: I could not find the answer in the document."""),
    ("human", """
    Question: {question}
    Context: {context}
    """)
])


# ─── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_vectorstore()

if "chunk_count" not in st.session_state:
    if st.session_state.vectorstore:
        try:
            st.session_state.chunk_count = st.session_state.vectorstore._collection.count()
        except:
            st.session_state.chunk_count = 0
    else:
        st.session_state.chunk_count = 0


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="title-text">RAG<br><span class="accent">Assistant</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">// powered by Groq + ChromaDB</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("⚡ Process & Embed"):
            with st.spinner("Processing PDF..."):
                progress = st.progress(0)
                progress.progress(20)

                vectorstore, chunk_count = process_pdf(uploaded_file)
                progress.progress(80)

                st.session_state.vectorstore = vectorstore
                st.session_state.chunk_count = chunk_count
                st.session_state.messages = []
                progress.progress(100)

            st.success(f"✅ Done! {chunk_count} chunks embedded.")

    st.markdown("---")

    if st.session_state.vectorstore:
        st.markdown(
            f'<div class="chunk-info">📦 {st.session_state.chunk_count} chunks in memory</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="status-box">🟢 Vector store ready</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.markdown('<div class="status-box">⚪ No document loaded yet</div>', unsafe_allow_html=True)


# ─── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown('<p class="title-text">Ask your <span class="accent">Document</span></p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">// upload a PDF → ask anything about it</p>', unsafe_allow_html=True)
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask something about your document..."):
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = get_retriever(st.session_state.vectorstore)
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                final_prompt = prompt_template.invoke(
                    {"context": context, "question": query}
                )

                llm = get_llm()
                response = llm.invoke(final_prompt)
                answer = response.content

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})