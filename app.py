from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.vectorstores import Pinecone as PC
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import BaseRetriever



app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Download embeddings
embeddings = download_hugging_face_embeddings()
print(f"Embeddings downloaded: {embeddings}")

# Define the index name
index_name = "medical-chatbot-2"

# Load the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
print(f"Docsearch initialized: {docsearch}")

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
print(f"Prompt template created: {PROMPT}")

# Define the language model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})
print(f"Language model initialized: {llm}")

# Define chain type kwargs
chain_type_kwargs = {"prompt": PROMPT}

retriever = docsearch.as_retriever(search_kwargs={"k": 2}
)

# Initialize the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    chain_type_kwargs=chain_type_kwargs,
    retriever=retriever,
    return_source_documents=True
)
print(f"QA chain initialized: {qa}")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8060, debug=True)
