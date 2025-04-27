# Import HuggingFaceEmbeddings to generate vector representations (embeddings) of text
from langchain_huggingface import HuggingFaceEmbeddings
# Import FAISS to create and manage an efficient vector store (similarity search engine)
from langchain_community.vectorstores import FAISS
# Import a PDF loader to extract text content from PDFs
from langchain_community.document_loaders import PDFMinerLoader
# Import a text splitter to divide documents into manageable chunks
from langchain_text_splitters import CharacterTextSplitter

# Define a class to manage the RAG (Retrieval Augmented Generation) process
class RAGManager:
    def __init__(self):
        # Initialize HuggingFaceEmbeddings to convert text to embeddings (vectors)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",              # Choose a lightweight model that's fast and good for semantic search
            model_kwargs={"device": "cpu"},             # Run on CPU (can switch to GPU if needed)
            encode_kwargs={"normalize_embeddings": True} # Normalize output embeddings for better cosine similarity search
        )
        
        # Initialize a text splitter to split documents into smaller, overlapping chunks
        self.text_splitter = CharacterTextSplitter(
            chunk_size=600,        # Each chunk is about 600 characters
            chunk_overlap=150,     # 150 characters overlap between chunks (helps preserve context across chunks)
            separator="\n\n"       # Split text on double newlines (natural paragraph breaks)
        )
        
        # Initialize the vectorstore object (will be assigned later)
        self.vectorstore = None

    # Method to load documents from a PDF file
    def load_documents(self, pdf_path: str):
        # Use PDFMinerLoader to extract text from the given PDF path
        docs = PDFMinerLoader(pdf_path).load()
        # Split extracted text into smaller chunks for better embedding and retrieval
        return self.text_splitter.split_documents(docs)

    # Method to create a vectorstore (FAISS index) from a list of documents
    def create_vectorstore(self, documents):
        # Add metadata to each document (optional: here tagging documents as medical)
        for doc in documents:
            doc.metadata = {"medical_content": True}

        # Build the FAISS vector store from the documents and their embeddings
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the FAISS index locally (to avoid recomputation later)
        self.vectorstore.save_local("faiss_index")

    # Method to load an already saved vectorstore (instead of recreating)
    def load_vectorstore(self, path="faiss_index"):
        # Load FAISS index from the given path using the same embeddings model
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True # Allow full deserialization (needed for FAISS internal data)
        )

    # Method to perform a search query against the vectorstore
    def query(self, question: str):
        # Raise an error if vectorstore is not loaded yet
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded yet.")
        
        # Enrich the user query by adding keywords (boosts retrieval quality for medical context)
        query = f"{question} symptoms treatment dosage"
        
        # Perform a similarity search on the vectorstore
        results = self.vectorstore.similarity_search(
            query,
            k=4, # Retrieve top 4 similar documents
            filter=lambda d: hasattr(d, "metadata") # Optional: Only consider documents with metadata
        )
        
        # Return the results in a clean dictionary format (content + metadata)
        return [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results]
