from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=1,
    load_all_available_meta=True,
)


doc = retriever.invoke("deep learning")

for i in doc:
    
    print(i.page_content[:500])
    print()
    print()
    print()
   