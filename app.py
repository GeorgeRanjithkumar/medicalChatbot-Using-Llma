from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load credentials
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Embeddings and Vector Store
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize LLM and RAG Chain
chatModel = ChatOpenAI(model="gpt-4o")

# Ensure your system_prompt in src.prompt includes {context}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "No message received", 400
    
    print(f"User Input: {msg}")
    
    try:
        # RAG Logic
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm sorry, I couldn't find an answer.")
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request.", 500

if __name__ == '__main__':
    # Running on 8080 as per your previous setup
    app.run(host="0.0.0.0", port=8080, debug=True)