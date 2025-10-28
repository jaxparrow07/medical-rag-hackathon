# Configuration Guide

## Quick Start

All configuration is centralized in `src/config.py`. You can easily switch between different performance presets and customize every aspect of the RAG system.

## 🎯 Performance Presets

### Quick Preset Selection

In `src/config.py`, change this line:

```python
ACTIVE_PRESET = 'maximum_accuracy'  # Options: 'maximum_accuracy', 'balanced', 'maximum_speed'
```

### Available Presets

#### 1. Maximum Accuracy (Recommended for Medical Use)
```python
ACTIVE_PRESET = 'maximum_accuracy'
```
- **Best for**: Medical Q&A, critical information
- **Precision**: float32 (full precision)
- **Speed**: ~300-400 docs/sec
- **VRAM**: ~2.5GB
- **Deterministic**: Yes
- **TF32**: Disabled

#### 2. Balanced
```python
ACTIVE_PRESET = 'balanced'
```
- **Best for**: General use, good balance
- **Precision**: float32
- **Speed**: ~500-600 docs/sec
- **VRAM**: ~2.5GB
- **Deterministic**: Yes
- **TF32**: Disabled

#### 3. Maximum Speed
```python
ACTIVE_PRESET = 'maximum_speed'
```
- **Best for**: Large datasets, fast prototyping
- **Precision**: float16 (half precision)
- **Speed**: ~800-1000 docs/sec
- **VRAM**: ~1.5GB
- **Deterministic**: No (faster)
- **TF32**: Enabled (RTX 40 series)

## 📝 System Prompts

### Switch Prompts

Change the active prompt in `src/config.py`:

```python
SYSTEM_PROMPTS['active'] = 'main_prompt'  # Options: 'main_prompt', 'concise_prompt', 'detailed_prompt'
```

### Available Prompts

**1. main_prompt** (Default)
- Clear, accessible language
- Strict anti-hallucination rules
- Good for general medical Q&A

**2. concise_prompt**
- Shorter, more direct
- Minimal verbosity
- Good for quick answers

**3. detailed_prompt**
- More comprehensive
- Medical terminology with explanations
- Good for detailed medical information

### Custom Prompts

Add your own prompt template:

```python
SYSTEM_PROMPTS = {
    # ... existing prompts ...
    
    'custom_prompt': """Your custom prompt here.
    
Context:
{context}

Question: {query}

Answer:""",
    
    'active': 'custom_prompt'  # Use your custom prompt
}
```

## 🤖 LLM Configuration

### Change Provider

```python
LLM_CONFIG = {
    'provider': 'openrouter',  # or 'gemini'
    # ...
}
```

### Change Model

```python
LLM_CONFIG = {
    'openrouter_model': 'deepseek/deepseek-r1',  # or other OpenRouter models
    'gemini_model': 'gemini-2.0-flash-exp',
    # ...
}
```

### Fine-tune Generation

```python
LLM_CONFIG = {
    'temperature': 0.1,  # Lower = more factual (0.0-1.0)
    'top_p': 0.8,        # Lower = more focused (0.0-1.0)
    'top_k': 20,         # Lower = more focused (1-100)
    'max_output_tokens': 1024,
}
```

**Temperature Guide:**
- `0.0-0.3`: Factual, deterministic (medical use)
- `0.4-0.7`: Balanced
- `0.8-1.0`: Creative, varied

## 🔍 Embedding Configuration

### Change Embedding Model

```python
EMBEDDING_CONFIG = {
    'model_name': 'BAAI/bge-base-en-v1.5',  # Options below
    # ...
}
```

**Available Models:**
- `BAAI/bge-base-en-v1.5` - Best overall (335MB)
- `sentence-transformers/all-MiniLM-L6-v2` - Fastest (80MB)
- `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` - Medical-specific

### Manual Precision Settings

```python
EMBEDDING_CONFIG = {
    'dtype': 'float32',              # 'float32' or 'float16'
    'use_mixed_precision': False,    # True for speed
    'batch_size': 16,                # Adjust based on VRAM
    'enable_cudnn_benchmark': False, # True for speed
    'deterministic': True,           # False for speed
    'allow_tf32': False,             # True for RTX 30/40 speed
    'matmul_precision': 'highest',   # 'highest', 'high', 'medium'
}
```

## 🔧 Retrieval Settings

```python
RETRIEVAL_CONFIG = {
    'top_k': 5,                    # Number of contexts (3-10)
    'similarity_threshold': 0.3,   # Minimum score (0.0-1.0)
    'rerank': False,               # Enable for better accuracy
    'diversity_penalty': 0.0,      # Penalize similar results
}
```

## 🐛 Debug Settings

```python
DEBUG_CONFIG = {
    'verbose': True,              # Print initialization info
    'log_embeddings': False,      # Log embedding operations
    'log_retrievals': True,       # Log search results
    'log_llm_calls': True,        # Log LLM API calls
    'save_failed_queries': True,  # Save errors for debugging
}
```

## 📊 Benchmarking

### Enable Benchmarking

```python
BENCHMARK_CONFIG = {
    'enabled': True,  # Set to True to enable
    # ...
}
```

### Run Benchmark

```bash
python benchmark.py
```

This will test all three presets and show:
- Latency breakdown (embed, search, generation)
- Average confidence scores
- Recommendations

### Custom Benchmark Queries

```python
BENCHMARK_CONFIG = {
    'enabled': True,
    'test_queries': [
        "Your custom query 1",
        "Your custom query 2",
        "Your custom query 3",
    ],
}
```

## 💾 Hardware Configuration

Document your hardware (for reference):

```python
HARDWARE_CONFIG = {
    'device': 'cuda',  # Auto-detected
    'gpu_name': 'RTX 4050',
    'vram_gb': 6,
    'ram_gb': 16,
    'cpu': 'Ryzen 7 7000',
}
```

## 🎓 Common Configurations

### Maximum Accuracy (Medical Use)

```python
ACTIVE_PRESET = 'maximum_accuracy'
SYSTEM_PROMPTS['active'] = 'main_prompt'
LLM_CONFIG['temperature'] = 0.05
RETRIEVAL_CONFIG['top_k'] = 5
DEBUG_CONFIG['verbose'] = True
```

### Fast Prototyping

```python
ACTIVE_PRESET = 'maximum_speed'
SYSTEM_PROMPTS['active'] = 'concise_prompt'
LLM_CONFIG['temperature'] = 0.1
RETRIEVAL_CONFIG['top_k'] = 3
DEBUG_CONFIG['verbose'] = False
```

### Detailed Medical Research

```python
ACTIVE_PRESET = 'maximum_accuracy'
SYSTEM_PROMPTS['active'] = 'detailed_prompt'
LLM_CONFIG['temperature'] = 0.1
RETRIEVAL_CONFIG['top_k'] = 10
RETRIEVAL_CONFIG['rerank'] = True
```

## 🔄 Switching Models

### Try Different LLMs

```python
# DeepSeek R1 (FREE, accurate)
LLM_CONFIG['provider'] = 'openrouter'
LLM_CONFIG['openrouter_model'] = 'deepseek/deepseek-r1'

# Llama 3.2 (FREE, fast)
LLM_CONFIG['openrouter_model'] = 'meta-llama/llama-3.2-3b-instruct:free'

# Gemini (Google, fast)
LLM_CONFIG['provider'] = 'gemini'
LLM_CONFIG['gemini_model'] = 'gemini-2.0-flash-exp'

# Claude 3.5 Sonnet (paid, excellent)
LLM_CONFIG['provider'] = 'openrouter'
LLM_CONFIG['openrouter_model'] = 'anthropic/claude-3.5-sonnet'
```

## 📈 Performance Tips

**For RTX 4050 (6GB VRAM):**

1. **Maximum Accuracy**: Use as-is
2. **Need more speed?**: Switch to `balanced` preset
3. **Processing large datasets?**: Use `maximum_speed` preset
4. **Running out of VRAM?**: Reduce `batch_size` to 8

**Batch Size Guidelines:**
- 16: Maximum accuracy, stable
- 32: Balanced
- 64: Maximum speed (if VRAM allows)

## 🧪 Testing Configuration

After changing config:

```bash
# Test embedding
python -c "from src.embedding import LocalEmbedder; e = LocalEmbedder(); print('OK')"

# Test generation
python test_deepseek.py

# Run benchmark
python benchmark.py
```

## ⚡ Quick Reference

| Setting | Accuracy | Balanced | Speed |
|---------|----------|----------|-------|
| dtype | float32 | float32 | float16 |
| batch_size | 16 | 32 | 64 |
| cudnn_benchmark | False | False | True |
| allow_tf32 | False | False | True |
| temperature | 0.05 | 0.1 | 0.1 |

---

Need help? Check `IMPLEMENTATION_SUMMARY.md` for technical details!
