# Document-Based Question Answering Chatbot

This project implements a Python-based chatbot that answers user questions using open-source Large Language Models (LLMs). The chatbot accesses information from a provided PDF document to answer questions accurately.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Model Architecture](#model-architecture)
- [Database Choice](#database-choice)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

## Overview

The chatbot prioritizes factual accuracy, understands the context of user questions, and indicates gracefully when information is not found within the document.

## Features

- **Factual Accuracy**: Answers are derived directly from the provided document.
- **Context Understanding**: Utilizes relevant information from the document to answer questions.
- **Knowledge of Unknown**: Indicates when the information is not found within the document.

## Requirements

- Python 3.7 or higher
- The following Python packages:
  - PyMuPDF (`fitz`)
  - Huggingface Transformers
  - Sentence Transformers
  - Faiss
  - Flask

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/ritikhandelwal/LLM_Projects
cd pdf_chatbot
Create a virtual environment and activate it:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Download the PDF file (NVIDIAAn.pdf) and place it in the project directory.
Usage
Run the Flask server:
bash
Copy code
python chatbot.py
Ask questions via a POST request to http://127.0.0.1:5000/ask with a JSON payload:
json
Copy code
{
  "question": "Your question here"
}
Technical Details
Text Processing
Text Extraction: Uses PyMuPDF to extract text from the PDF document.
Cleaning and Normalization: Lowercases the text and removes punctuation.
Chunking: Splits the text into chunks of 512 tokens for processing.
Embedding Generation
Model: Uses the paraphrase-MiniLM-L6-v2 model from Sentence Transformers to generate embeddings for text chunks.
Vector Store: Uses Faiss for efficient similarity searches.
Question Answering
Model: Uses distilbert-base-uncased-distilled-squad from Huggingface Transformers for question answering.
Answer Selection: Finds the most relevant text chunks using Faiss, then extracts answers using the DistilBERT model.
Model Architecture
Text Extraction and Preprocessing: Extract text from the PDF, clean and normalize it, then split it into chunks.
Embedding Generation: Generate embeddings for the text chunks using the Sentence Transformer model.
Indexing with Faiss: Create a Faiss index of the embeddings for efficient similarity searches.
Question Answering: Use DistilBERT to find answers in the most relevant text chunks.
Database Choice
Vector Database: Faiss is used for managing embeddings and performing similarity searches due to its efficiency in handling high-dimensional data.
Limitations and Future Improvements
Limitations:

The chatbot may still provide inaccurate answers to questions not covered by the document.
Processing very large documents may require optimization for efficiency.
Future Improvements:

Integrate more sophisticated models or multiple documents.
Implement better handling for identifying irrelevant questions.
Explore hybrid database approaches for more complex data management needs.
