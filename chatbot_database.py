import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import fitz  # PyMuPDF
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from flask import Flask, request, jsonify

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Clean and normalize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess text: split into chunks
def split_text_into_chunks(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Initialize the tokenizer and models
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer=tokenizer)

# Load a pre-trained SentenceTransformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for text chunks
def generate_embeddings(text_chunks):
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Flask app setup
app = Flask(__name__)

# Load and preprocess the document
document_path = "NVIDIAAn.pdf"
document_text = clean_text(extract_text_from_pdf(document_path))
text_chunks = split_text_into_chunks(document_text)
embeddings = generate_embeddings(text_chunks)

# Initialize Faiss index and add embeddings
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Function to query the index
def query_index(query, k=5):
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# Function to get the answer from the model
def get_answer_from_model(question, context):
    result = qa_model(question=question, context=context)
    return result['answer'], result['score']

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "Question not provided"}), 400

    # Find relevant text chunks using Faiss
    indices = query_index(question, k=10)  # Increase k to retrieve more chunks
    relevant_chunks = [text_chunks[i] for i in indices]

    # Generate answers from the model
    answers = []
    for chunk in relevant_chunks:
        try:
            answer, score = get_answer_from_model(question, chunk)
            answers.append((answer, score))
        except Exception as e:
            continue

    if not answers:
        return jsonify({"answer": "I could not find the answer in the document."})

    # Sort answers by score and return the best one
    best_answer = sorted(answers, key=lambda x: x[1], reverse=True)[0][0]
    return jsonify({"answer": best_answer})

if __name__ == '__main__':
    app.run(debug=True)
