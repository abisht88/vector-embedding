import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    GithubFileLoader,
    ConfluenceLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from jina_embeddings import JinaEmbeddings
from langchain_chroma import Chroma
from github_repos.github_repos import GITHUB_REPOS

load_dotenv()

# ==========================================
# CONFIG
# ==========================================

docs_dir = "./rag_pipeline_docs"
persist_db_dir = "chroma_db"

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
        access_token=os.getenv.get("GITHUB_TOKEN"),
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

print("Loading Confluence pages...")

confluence_loader = ConfluenceLoader(
    url="https://ashutoshbisht88.atlassian.net/wiki",
    username=os.getenv.get("CONFLUENCE_API_TOKEN"),
    api_key=os.getenv.get("CONFLUENCE_USER"),
    space_key="RAGHACK",
    cloud=True
)

try:
    # 1. Fetch all documents from the space
    confluence_documents = confluence_loader.load()

    print(f"--- Found {len(confluence_documents)} pages in space 'RAGHACK' ---\n")

    # 2. Iterate and print clear metadata anchors
    for idx, doc in enumerate(confluence_documents, 1):
        title = doc.metadata.get("title", "Untitled Page")
        source_url = doc.metadata.get("source", "No URL available")

        print(f"[{idx}] {title}")
        print(f"    🔗 Link: {source_url}")
        print("-" * 40)

    print(f"Loaded {len(confluence_documents)} Confluence pages")

    all_documents.extend(confluence_documents)
except Exception as e:
    print(f"Failed to fetch pages: {e}")



# 1. Load the Investopedia webpage
url = "https://www.investopedia.com/terms/q/quantitative-trading.asp"
loader = WebBaseLoader(
    web_path=url,
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
)
raw_documents = loader.load()

print(f"Loaded {len(raw_documents)} investopedia pages...")

all_documents.extend(raw_documents)

# # 2. Configure the recursive text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,       # Targets ~1000 characters per chunk
#     chunk_overlap=200,     # Overlaps 200 characters to prevent cutting off context
#     length_function=len,
#     separators=["\n\n", "\n", " ", ""]  # Split priorities
# )
#
# # 3. Split the loaded document into clean chunks
# chunked_documents = text_splitter.split_documents(raw_documents)
#
# print(f"📄 Original Documents: {len(raw_documents)}")
# print(f"🧩 Total Text Chunks Created: {len(chunked_documents)}\n")
#
# # 4. Preview the first 2 chunks
# for idx, chunk in enumerate(chunked_documents[:2], 1):
#     print(f"--- Chunk {idx} (Length: {len(chunk.page_content)}) ---")
#     print(chunk.page_content.strip())
#     print(f"Metadata: {chunk.metadata}\n")

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

embeddings = JinaEmbeddings(
    api_key=os.environ["JINA_API_KEY"],
    model="jina-embeddings-v5"
)

# ==========================================
# CREATE CHROMA DB
# ==========================================

print("Creating Chroma Vector Database...")

BATCH_SIZE = 50

vector_store = Chroma(
    persist_directory=persist_db_dir,
    embedding_function=embeddings
)

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    print(f"Processing Batch : {i}")
    seen_ids = set()
    dedup_batch = []
    custom_ids = []
    for doc in batch:
        # Encode string to bytes for hashing
        text_bytes = doc.page_content.encode('utf-8')
        # Create a hex string hash
        content_hash = hashlib.md5(text_bytes).hexdigest()
        if content_hash in seen_ids:
            continue
        seen_ids.add(content_hash)
        dedup_batch.append(doc)
        custom_ids.append(content_hash)

    # 3. Add documents with their unique hashes
    vector_store.add_documents(documents=dedup_batch, ids=custom_ids)

print("\n======================================")
print("Embedding completed successfully!")
print(f"Documents Indexed : {len(all_documents)}")
print(f"Chunks Created    : {len(documents)}")
print(f"Vector DB         : {os.path.abspath(persist_db_dir)}")
print("======================================")