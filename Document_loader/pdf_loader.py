from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
doc = PyPDFLoader("Document_loader/Data-Science.pdf")
data = doc.load()
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizes the text."),
        ("human", "{data}"),
    ]
)

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
prompt = template.format_prompt(data=data[0].page_content)
response = model.invoke(prompt)
print(response.content)