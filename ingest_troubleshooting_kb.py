"""One-off ingestion of troubleshooting_kb.md into a dedicated persistent
Chroma collection, so the chatbot can use it to diagnose errors seen in the
dummy service's logs.

Usage: ./venv/bin/python ingest_troubleshooting_kb.py
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

SOURCE_FILE = Path(__file__).parent / "troubleshooting_kb.md"
PERSIST_DIR = "./chroma_db_troubleshooting"
COLLECTION_NAME = "troubleshooting_kb"


def main():
    text = SOURCE_FILE.read_text()

    # Split by ## headers first so each error's Error/Cause/Solution block
    # stays together as one chunk, then fall back to size-based splitting
    # only if a section is unusually long.
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    sections = header_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = []
    for section in sections:
        title = section.metadata.get("section", "")
        for chunk in char_splitter.split_text(section.page_content):
            docs.append(Document(page_content=chunk, metadata={"source": "troubleshooting_kb.md", "section": title}))

    print(f"Split into {len(docs)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_store.add_documents(docs)
    print("Done.")


if __name__ == "__main__":
    main()
