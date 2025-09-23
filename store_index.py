import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -----------------------------
# Functions
# -----------------------------
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f'You have {len(documents)} documents in your data')
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f'Now you have {len(text_chunks)} chunks of data')
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# -----------------------------
# Pinecone index handling
# -----------------------------
def create_or_get_index(index_name, dimension=384):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes()
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")
    return pc

# -----------------------------
# Main execution (only runs when executed directly)
# -----------------------------
if __name__ == "__main__":
    # Load & split documents
    extracted_data = load_pdf_file("Data/")
    text_chunks = text_split(extracted_data)
    
    # Load embeddings
    embeddings = download_hugging_face_embeddings()
    
    # Create or connect to Pinecone index
    index_name = "agriculture-chatbot"
    pc = create_or_get_index(index_name)
    
    # Upsert documents to Pinecone
    from langchain_pinecone import PineconeVectorStore
    docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
    print("Documents have been upserted to Pinecone.")
