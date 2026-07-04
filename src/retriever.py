"""
Retrieval module: find the most relevant chunks for a user query.
"""
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document


def get_retriever(vectorstore: Chroma, top_k: int = 5):
    """Create a retriever from the vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve_documents(
    retriever,
    query: str,
    top_k: int = 5,
    section_filter: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve relevant documents for a query with optional metadata filtering.
    """
    if section_filter:
        docs = retriever.vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"section": section_filter},
        )
    else:
        retriever.search_kwargs = {"k": top_k}
        docs = retriever.invoke(query)
    return docs


def format_retrieved_context(docs: List[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        section = doc.metadata.get("section", "General")
        source = doc.metadata.get("source", "Knowledge Base")
        parts.append(
            f"[Source {i}] Section: {section} | File: {source}\n"
            f"{doc.page_content.strip()}\n"
        )
    return "\n\n".join(parts)
