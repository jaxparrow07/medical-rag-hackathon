# Medical RAG System

A Retrieval-Augmented Generation (RAG) system for medical question answering using local embeddings and Gemini 2.5 Flash.

## Features

- 📚 PDF document processing and chunking
- 🔍 Local embeddings with BGE-base model
- 💾 Persistent ChromaDB vector storage
- 🤖 Answer generation with Gemini 2.5 Flash
- 📝 Automatic citation generation
- ⚡ Query spell correction and expansion
- 🌐 FastAPI REST API

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API key:

```powershell
copy .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Add PDF Documents

Place your medical PDF files in the `data/raw_pdfs/` directory.

### 4. Setup Vector Database

Run the setup script to process PDFs and create the vector database:

```powershell
python setup_database.py
```

This will:
- Extract text from all PDFs
- Generate embeddings using BGE-base model
- Store everything in ChromaDB at `data/vectordb/`

## Usage

### Interactive Mode (Recommended)

Run the main file to enter interactive mode:

```powershell
python main.py
```

Then type your medical questions and press Enter. Type `exit` to quit.

### Test Sample Queries

```powershell
python main.py --test
```

### Single Query

```powershell
python main.py "What are the symptoms of diabetes?"
```

### API Server

Start the FastAPI server:

```powershell
python src/api.py
```

Then make POST requests to `http://localhost:8000/query`:

```json
{
  "query": "What are the symptoms of diabetes?",
  "top_k": 5
}
```

## Project Structure

```
rag-model/
├── main.py                 # Main entry point for queries
├── setup_database.py       # Database setup script
├── test_rag.py            # API testing script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/
│   ├── raw_pdfs/         # Place your PDF files here
│   ├── processed/        # Processed data
│   └── vectordb/         # ChromaDB storage (auto-created)
└── src/
    ├── api.py            # FastAPI application
    ├── embedding.py      # Local embedding model
    ├── vectordb.py       # ChromaDB vector store
    ├── ingestion.py      # PDF processing
    ├── query_processor.py # Query correction/expansion
    ├── generation.py     # Gemini answer generation
    └── config.py         # Configuration settings
```

## Models Used

- **Embeddings**: BAAI/bge-base-en-v1.5 (335MB, runs locally)
- **LLM**: Gemini 2.5 Flash (API-based)
- **Vector DB**: ChromaDB (embedded, no separate service needed)

## Example Queries

```
What are the symptoms of diabetes?
How is hypertension treated?
What causes pneumonia?
What are the risk factors for heart disease?
How to diagnose asthma?
```

## Notes

- ChromaDB runs embedded - no separate database service needed
- First run will download the BGE embedding model (~335MB)
- All data persists in `data/vectordb/` directory
- Answers include citations from source documents
