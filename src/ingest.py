"""
Document ingestion: load, chunk, embed, and index the knowledge base document.
Uses langchain-chroma and langchain-huggingface (no deprecated community wrappers).
"""
import os
import shutil
from typing import Tuple, List
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Persistent directory for ChromaDB
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bvrit_kb"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get free local embeddings using sentence-transformers (no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_document(filepath: str) -> List[Document]:
    """Load a .docx file and return documents with source metadata."""
    loader = Docx2txtLoader(filepath)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(filepath)
    return documents


def extract_section_from_chunk(chunk_text: str) -> str:
    """Extract section heading directly from chunk text."""
    section_markers = {
        "About BVRITH": [
            "SECTION 1", "About BVRITH", "ABOUT BVRITH", "About BVRIT",
            "Full name: BVRIT HYDERABAD", "BVRITH is the youngest",
            "History and Milestones", "Vision:", "Mission — BVRITH",
            "Core Values", "National recognitions",
        ],
        "Departments": [
            "SECTION 2", "DEPARTMENTS", "Departments",
            "BVRITH currently runs", "UG B.Tech departments",
            "Postgraduate (M.Tech) programs", "Ph.D Research Centres",
            "Honor Degree Program", "Department-level",
        ],
        "Admissions": [
            "SECTION 3", "ADMISSIONS", "Admissions",
            "Eligibility for B.Tech", "Admission routes",
            "Category A seats", "Category B seats",
            "How to apply", "Entrance exam codes",
            "EAMCET / ECET College Code", "PG (M.Tech) admission",
            "Rank reference",
        ],
        "Fee Structure": [
            "SECTION 4", "FEE STRUCTURE", "Fee Structure",
            "Tuition Fee", "NBA Fee", "JNTUH/Misc. Fee",
            "2025 Batch:", "2024 Batch:", "2023 Batch:",
            "2022 Batch:", "2021 Batch:", "2020 Batch:",
            "INR 1,20,000", "INR 90,000",
        ],
        "Placements": [
            "SECTION 5", "PLACEMENTS", "Placements",
            "Placement records", "Total students placed",
            "Highest package", "High-volume recruiters",
            "2021–2025 Batch", "2020–2024 Batch", "2019–2023 Batch",
            "2018–2022 Batch", "2017–2021", "2016–2020",
            "Recruiters that have hired", "Placement Team",
        ],
        "Campus & Facilities": [
            "SECTION 6", "CAMPUS & FACILITIES", "Campus & Facilities",
            "Location:", "Library", "Library hours",
            "Other campus facilities", "Student life and clubs",
            "Differentiator centres",
        ],
        "Faculty": [
            "SECTION 7", "FACULTY", "Faculty",
            "Faculty information", "Institutional leadership",
            "Research at BVRITH",
        ],
        "Contact": [
            "SECTION 8", "CONTACT", "Contact",
            "BVRIT HYDERABAD College of Engineering for Women",
            "Address:", "Phone:", "Admissions contact",
            "Email:", "Social media:", "Parent society",
            "Sri Vishnu Educational Society",
        ],
    }

    for section_name, keywords in section_markers.items():
        for keyword in keywords:
            if keyword in chunk_text:
                return section_name
    return "General"


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split documents into chunks; tag each chunk with its section."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\nSECTION ", "\nSection ", "\n\n\n", "\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["section"] = extract_section_from_chunk(chunk.page_content)
    return chunks


def index_documents(chunks: List[Document]) -> Chroma:
    """Embed and index document chunks into ChromaDB."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def load_and_index_document(
    filepath: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Tuple[Chroma, int]:
    """
    Full pipeline: load → split → embed → index.
    Reloads the existing index if it has data; re-indexes otherwise.
    """
    embeddings = get_embeddings()

    # Try loading existing index
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        chunk_count = vectorstore._collection.count()
        if chunk_count > 0:
            return vectorstore, chunk_count
        # Empty collection — wipe and re-index
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # Fresh index
    documents = load_document(filepath)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vectorstore = index_documents(chunks)
    return vectorstore, len(chunks)
