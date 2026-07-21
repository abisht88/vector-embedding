import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Use an 'r' before the string to handle Windows backslashes properly
docs_dir = r"./rag_pipeline_docs"
persist_db_dir = "./chroma_db"

print("Loading documents...")
loader = PyPDFDirectoryLoader(docs_dir)
raw_documents = loader.load()

if not raw_documents:
    print(f"Warning: No PDF files found in {docs_dir}")
else:
    print(f"Loaded {len(raw_documents)} pages. Splitting text...")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)

    # 3. Use the updated OllamaEmbeddings from 'langchain-ollama'
    print("Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Use Chroma from the dedicated 'langchain-chroma' package
    print("Generating embeddings and creating local Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_db_dir
    )

    print(f"Success! Vector database created inside: {os.path.abspath(persist_db_dir)}")
