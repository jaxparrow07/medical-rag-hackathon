# Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### Step 1: Get API Key
👉 **OpenRouter** (Recommended - FREE): https://openrouter.ai/keys  
👉 **Gemini** (Alternative): https://makersuite.google.com/app/apikey

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Setup Environment
```powershell
# Copy the example env file
copy .env.example .env

# Edit .env and add your API key:
# For OpenRouter (recommended):
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# OR for Gemini:
GEMINI_API_KEY=your-key-here
```

**Note:** DeepSeek R1 via OpenRouter is the default (FREE tier available).  
To use Gemini instead, edit `src/api.py` and change the provider.

## Step 3: Add Your PDF Files
Place your medical PDF files in `data/raw_pdfs/` folder

## Step 4: Build Vector Database
```powershell
python setup_database.py
```

This will process PDFs and create embeddings (first time downloads ~335MB model).

## Step 5: Query the System
```powershell
# Interactive mode (ask multiple questions)
python main.py

# Test with sample queries
python main.py --test

# Single query
python main.py "What are the symptoms of diabetes?"
```

## Alternative: Use API Server
```powershell
# Start the server
python src/api.py

# In another terminal, test it
python test_rag.py
```

## Quick Test (Without PDFs)
```powershell
# Test DeepSeek integration directly
python test_deepseek.py
```

## Troubleshooting

**Problem**: "OPENROUTER_API_KEY not found"  
**Solution**: Make sure you created `.env` file with `OPENROUTER_API_KEY=your-key`

**Problem**: "GEMINI_API_KEY not found" (if using Gemini)  
**Solution**: Make sure you created `.env` file with your API key

**Problem**: "Vector database is empty"  
**Solution**: Run `python setup_database.py` first

**Problem**: "No PDFs found"  
**Solution**: Add PDF files to `data/raw_pdfs/` directory

**Problem**: ChromaDB errors  
**Solution**: Delete `data/vectordb/` folder and run `setup_database.py` again

**Problem**: Import errors for `openai` or `google-generativeai`  
**Solution**: Run `pip install -r requirements.txt` again

## 🔄 Switching Providers

### To use Gemini instead of OpenRouter:

1. Get Gemini API key from https://makersuite.google.com/app/apikey
2. Add to `.env`: `GEMINI_API_KEY=your-key`
3. Edit `src/api.py`, change:
```python
citation_gen = CitationGenerator(
    model="gemini-2.0-flash-exp",
    provider="gemini"
)
```

### To use other OpenRouter models:

```python
# In src/api.py:
citation_gen = CitationGenerator(
    model="meta-llama/llama-3.1-70b-instruct:free",  # Change model
    provider="openrouter"
)
```

See `LLM_PROVIDERS.md` for full list of available models.

## 📚 Next Steps

- Read `README.md` for full documentation
- Check `LLM_PROVIDERS.md` for provider comparison
- See `IMPLEMENTATION_SUMMARY.md` for technical details
