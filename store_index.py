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
# 1. Load agriculture PDFs
# -----------------------------
def load_pdf_file(data) :
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f'You have {len(documents)} documents in your data')
    return documents

#split the data into text chunks
extracted_data = load_pdf_file(data="Data/")
def text_split(extracted_data) :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f'Now you have {len(text_chunks)} chunks of data')
    return text_chunks

text_chunks = text_split(extracted_data)

#Download the embedding model from HuggingFace

def download_hugging_face_embeddings() :
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "agriculture-chatbot"
pc.create_index(name=index_name,
                dimension=384, 
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud="aws", region="us-east-1")
        )

#Embed each chunk and upsert the embeddings to Pinecone
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)