# Quick Start Guide

## Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

## Step 2: Setup Environment
```powershell
# Copy the example env file
copy .env.example .env

# Edit .env and add your Gemini API key
# Get it from: https://makersuite.google.com/app/apikey
```

## Step 3: Add Your PDF Files
Place your medical PDF files in `data/raw_pdfs/` folder

## Step 4: Build Vector Database
```powershell
python setup_database.py
```

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

## Troubleshooting

**Problem**: "GEMINI_API_KEY not found"
**Solution**: Make sure you created `.env` file with your API key

**Problem**: "Vector database is empty"
**Solution**: Run `python setup_database.py` first

**Problem**: "No PDFs found"
**Solution**: Add PDF files to `data/raw_pdfs/` directory

**Problem**: ChromaDB errors
**Solution**: Delete `data/vectordb/` folder and run `setup_database.py` again
