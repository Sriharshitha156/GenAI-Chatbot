"""
Run the full 8-dimension evaluation suite for the BVRIT FAQ chatbot.
"""
import os
import dotenv
dotenv.load_dotenv()

from src.ingest import load_and_index_document, load_document
from src.retriever import get_retriever
from src.evaluation import run_evaluation

# Load the index
print("Loading vector store...")
vectorstore, chunk_count = load_and_index_document('bvrit_knowledge_base.docx')
retriever = get_retriever(vectorstore)
print(f"Loaded {chunk_count} chunks")

# Load document text for test generation
docs = load_document('bvrit_knowledge_base.docx')
kb_text = docs[0].page_content
print(f"Knowledge base text length: {len(kb_text)} chars")

# Run evaluation
print("\n" + "=" * 60)
print("STARTING EVALUATION")
print("=" * 60)
results, report = run_evaluation(retriever, kb_text, top_k=5)

# Save report to file
with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\nEvaluation report saved to evaluation_report.txt")