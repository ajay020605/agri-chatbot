from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain.chains import RefineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

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

# Prompt for the first chunk
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template=prompt_template
)

# Prompt for refining subsequent chunks
refine_prompt = PromptTemplate(
    input_variables=["document", "prev_response"],
    template="Refine the following answer based on this new document:\n\n{document}\n\nPrevious Answer: {prev_response}"
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=document_prompt)


# Create RAG chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# Helper function for greetings / short input
# -----------------------------
def handle_greetings_or_small_input(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    user_input_lower = user_input.strip().lower()

    if user_input_lower in greetings:
        return "Hello! How can I help you with agriculture today?"
    
    if len(user_input.strip()) < 3:  # very short inputs
        return "Could you please provide more details so I can help you better?"

    return None  # go through RAG if input is normal

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    print(f"User input: {msg}")

    # Step 1: Check for greetings or very short/unrelated input
    pre_response = handle_greetings_or_small_input(msg)
    if pre_response:
        print(f"Bot response (greeting/small input): {pre_response}")
        return pre_response

    # Step 2: Otherwise, go through RAG chain
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "Sorry, I couldn't find an answer for that.")
    print(f"Bot response: {answer}")

    return str(answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
