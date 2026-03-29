from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

url = "https://www.zameen.com/"
data = WebBaseLoader(url)
doc = data.load()
# print(len(doc))
print(doc[0].page_content)