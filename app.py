from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain as create_stuff_documents
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMENIE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMENIE_API_KEY"] = GEMENIE_API_KEY

# -----------------------------
# Load embeddings
# -----------------------------
embeddings = download_hugging_face_embeddings()

# -----------------------------
# Connect to existing Pinecone index
# -----------------------------
index_name = "agriculture-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -----------------------------
# LLM and Prompt Setup
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GEMENIE_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        prompt_template,
    ]
)

from langchain.chains.combine_documents import AnalyzeDocumentChain, RefineDocumentsChain

# Step 1: Analyze each chunk individually with the LLM and your prompt
analyze_chain = AnalyzeDocumentChain(llm=llm, prompt=prompt)

# Step 2: Refine chain combines all chunk-level analyses into a coherent answer
question_answer_chain = RefineDocumentsChain(
    llm_chain=analyze_chain,
    refine_prompt=prompt  # can be same as your main prompt
)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    print(f"Input: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
