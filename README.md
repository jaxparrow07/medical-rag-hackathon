# Medical RAG System

A Retrieval-Augmented Generation (RAG) system for medical question answering using local embeddings and advanced LLMs (DeepSeek R1 or Gemini).

## Features

- 📚 PDF document processing and chunking
- 🔍 Local embeddings with BGE-base model
- 💾 Persistent ChromaDB vector storage
- 🤖 Multiple LLM providers:
  - **DeepSeek R1** via OpenRouter (FREE tier available)
  - **Gemini 2.0 Flash** (Google AI)
- 📝 Automatic citation generation
- ⚡ Query spell correction and expansion
- 🌐 FastAPI REST API
- 🛡️ Anti-hallucination measures

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

Edit `.env` and add your API key(s):

**Option A: OpenRouter (Recommended - FREE DeepSeek R1)**
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
Get your key from: https://openrouter.ai/keys

**Option B: Gemini**
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your key from: https://makersuite.google.com/app/apikey

### 2b. Choose Your LLM Provider

In `src/api.py`, the default is set to OpenRouter with DeepSeek R1:

```python
# Default: OpenRouter + DeepSeek R1 (FREE)
citation_gen = CitationGenerator(
    model="deepseek/deepseek-r1-0528:free",
    provider="openrouter"
)
```

To use Gemini instead, change to:
```python
citation_gen = CitationGenerator(
    model="gemini-2.0-flash-exp",
    provider="gemini"
)
```

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
- **LLM Options**:
  - **DeepSeek R1** via OpenRouter (FREE tier, recommended)
  - **Gemini 2.0 Flash** via Google AI
- **Vector DB**: ChromaDB (embedded, no separate service needed)

## LLM Provider Comparison

| Feature | DeepSeek R1 (OpenRouter) | Gemini 2.0 Flash |
|---------|--------------------------|------------------|
| **Cost** | FREE tier available | FREE tier with limits |
| **Speed** | ~2-3s per response | ~1-2s per response |
| **Quality** | Excellent reasoning | Excellent general |
| **Setup** | Get key from openrouter.ai | Get key from Google AI |
| **Medical Focus** | Strong analytical | Strong general knowledge |

## Example Queries

```
What are the symptoms of diabetes?
How is hypertension treated?
What causes pneumonia?
What are the risk factors for heart disease?
How to diagnose asthma?
```

## Testing

### Test DeepSeek Integration

```powershell
python test_deepseek.py
```

This will test both OpenRouter (DeepSeek R1) and optionally Gemini if configured.

### Test API

```powershell
python test_rag.py
```

## Notes

- ChromaDB runs embedded - no separate database service needed
- First run will download the BGE embedding model (~335MB)
- All data persists in `data/vectordb/` directory
- Answers include citations from source documents
- **Anti-hallucination features**:
  - Strict prompt engineering to prevent extrapolation
  - Low temperature (0.1) for deterministic responses
  - Post-processing to remove source mentions
  - Confidence scoring based on answer certainty

## Why DeepSeek R1?

DeepSeek R1 is recommended because:
- ✅ **FREE tier** available on OpenRouter
- ✅ Strong **reasoning capabilities** for medical questions
- ✅ Better at following **strict instructions** (no hallucination)
- ✅ Good at **staying within context** boundaries
- ✅ Accessible via simple REST API
