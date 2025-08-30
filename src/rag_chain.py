from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_PATH = r"..\data\chroma"
MODEL_NAME = "llama2"

class FreeRAGChain:
    def __init__(self):
        self.embeddings = None
        self.db = None
        self.llm = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """Initialize embeddings, vector store, LLM, and QA chain"""
        print("Initializing RAG system...")
        
        # Use updated HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"Chroma database not found at {CHROMA_PATH}. "
                "Please run ingest.py first"
            )

        # Use updated Chroma import
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        self.llm = OllamaLLM(
            model=MODEL_NAME,
            verbose=False,
            temperature=0.1,
        )

        self.qa_chain = self._create_qa_chain()
        print("RAG system initialized")

    def _create_qa_chain(self):
        """Create the QA chain with a custom prompt"""
        template = """Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer that question."

Context: {context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str):
        """Query the RAG system"""
        print(f"Searching for: {question}")
        result = self.qa_chain({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]

        # Format sources
        source_info = []
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            source_info.append(f"[{i}] {os.path.basename(source)} (page {page})")

        return {
            "answer": answer,
            "sources": source_info
        }

def create_rag_chain():
    """Factory function to create RAG chain"""
    return FreeRAGChain()
