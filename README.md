# Chat with PDF 📄🤖

An AI-powered web app that lets you upload any PDF and ask questions about it using **Google Gemini AI**.

---

## Features
- Upload any PDF (textbooks, notes, research papers)
- Ask questions in plain English
- Get accurate answers from the document
- Remembers conversation context
- Clean Streamlit web interface
- 100% Free using Google Gemini API

---

## Tech Stack
| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Reading | PyMuPDF |
| AI Model | Google Gemini 1.5 Flash |
| Embeddings | TF-IDF (Scikit-learn) |
| Similarity Search | Cosine Similarity |

---

## Setup & Run

### Step 1 — Install Python
Download Python 3.10+ from https://python.org

### Step 2 — Install Dependencies
Open terminal in this folder and run:
```
pip install -r requirements.txt
```

### Step 3 — Get Free Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click **Create API Key**
4. Copy the key

### Step 4 — Run the App
```
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Step 5 — Use the App
1. Paste your API key in the sidebar
2. Upload a PDF file
3. Click **Process PDF**
4. Start asking questions!

---

## Project Structure
```
ChatWithPDF/
├── app.py              ← Main Streamlit UI
├── pdf_processor.py    ← PDF text extraction & chunking
├── embeddings.py       ← TF-IDF vector store & search
├── gemini_handler.py   ← Google Gemini API integration
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## How It Works (RAG Architecture)
```
PDF Upload
    ↓
Extract Text (PyMuPDF)
    ↓
Split into Chunks (1000 chars, 200 overlap)
    ↓
Convert to TF-IDF Vectors (Scikit-learn)
    ↓
User asks question
    ↓
Find most similar chunks (Cosine Similarity)
    ↓
Send question + chunks to Gemini API
    ↓
Display answer to user
```
