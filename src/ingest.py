import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = r"..\data\raw"
CHROMA_PATH = r"..\data\chroma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_documents():
    """Load documents from directory"""
    print("Loading documents...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")
    return documents

def split_text(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def save_to_chroma(chunks):
    """Save chunks to chroma vector database"""
    print("Creating embeddings and saving to chroma...")
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def main():
    print("Starting document ingestion...")
    documents = load_documents()
    if not documents:
        print("No documents found. Add PDF files to data/raw/ directory")
        return
    
    chunks = split_text(documents)
    save_to_chroma(chunks)
    print("Ingestion complete! Ready for querying.")

if __name__ == "__main__":
    main()
