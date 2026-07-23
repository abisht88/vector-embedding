import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    GithubFileLoader,
    ConfluenceLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from github_repos.github_repos import GITHUB_REPOS

load_dotenv()

# ==========================================
# CONFIG
# ==========================================

docs_dir = "./rag_pipeline_docs"
persist_db_dir = "./chroma_db"

all_documents = []

# ==========================================
# LOAD PDF DOCUMENTS
# ==========================================

print("Loading PDF documents...")

if os.path.exists(docs_dir):
    pdf_loader = PyPDFDirectoryLoader(docs_dir)
    pdf_documents = pdf_loader.load()

    print(f"Loaded {len(pdf_documents)} PDF pages")
    all_documents.extend(pdf_documents)

# ==========================================
# LOAD GITHUB DOCUMENTS
# ==========================================

print("Loading GitHub documents...")
github_documents = []

for repo in GITHUB_REPOS:
    print(f"\nLoading repository: {repo}")
    
    loader = GithubFileLoader(
        repo=repo,
        branch="master",
        access_token=os.getenv("GITHUB_TOKEN"),
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(
            (
                ".md",
                ".txt",
                ".rst",
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".ini",
                ".cfg",
                ".xml",
                ".java",
                ".py",
                ".js",
                ".ts",
                ".cpp",
                ".hpp",
                ".cs",
                ".go",
                ".sql",
                #".png",
                #".jpg",
                ".properties",
            )
        ),
    )
    
    try:
        docs = loader.load()

        for doc in docs:
            doc.metadata["repository"] = repo

        github_documents.extend(docs)

        print(f"Loaded {len(docs)} documents from {repo}")

    except Exception as e:
        print(f"Failed to load {repo}")
        print(e)


print(f"\nTotal GitHub documents: {len(github_documents)}")

all_documents.extend(github_documents)

# ==========================================
# LOAD CONFLUENCE DOCUMENTS
# ==========================================

# print("Loading Confluence pages...")
#
# confluence_loader = ConfluenceLoader(
#     url=os.getenv("CONFLUENCE_URL"),
#     username=os.getenv("CONFLUENCE_EMAIL"),
#     api_key=os.getenv("CONFLUENCE_API_TOKEN"),
#     space_key="ENG",          # Change accordingly
# )
#
# confluence_documents = confluence_loader.load()
#
# print(f"Loaded {len(confluence_documents)} Confluence pages")
#
# all_documents.extend(confluence_documents)

# ==========================================
# SPLIT DOCUMENTS
# ==========================================

print(f"\nTotal raw documents: {len(all_documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = text_splitter.split_documents(all_documents)

print(f"Created {len(documents)} chunks")

# ==========================================
# EMBEDDINGS
# ==========================================

print("Initializing embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ==========================================
# CREATE CHROMA DB
# ==========================================

print("Creating Chroma Vector Database...")

BATCH_SIZE = 50

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    print(f"Processing Batch : {i}")
    if i == 0:
        vector_store = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=persist_db_dir
        )
    else:
        vector_store.add_documents(documents=batch)

print("\n======================================")
print("Embedding completed successfully!")
print(f"Documents Indexed : {len(all_documents)}")
print(f"Chunks Created    : {len(documents)}")
print(f"Vector DB         : {os.path.abspath(persist_db_dir)}")
print("======================================")