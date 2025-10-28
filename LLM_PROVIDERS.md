# LLM Provider Guide

This RAG system supports multiple LLM providers. Choose based on your needs.

**Note:** OpenRouter integration uses the OpenAI Python SDK for a cleaner, more maintainable implementation.

## 🌟 Recommended: OpenRouter + DeepSeek R1

### Why DeepSeek R1?
- ✅ **FREE tier available** (rate limited but generous)
- ✅ **Excellent reasoning** - DeepSeek R1 excels at analytical tasks
- ✅ **Strong instruction following** - stays within context boundaries
- ✅ **Low hallucination rate** - doesn't make up information
- ✅ **Good for medical domain** - handles technical content well
- ✅ **OpenAI SDK compatible** - clean, simple integration

### Setup

1. Get API key from: https://openrouter.ai/keys
2. Add to `.env`:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   ```
3. In `src/api.py`, use:
   ```python
   citation_gen = CitationGenerator(
       model="deepseek/deepseek-r1",
       provider="openrouter"
   )
   ```

### Other OpenRouter Models

You can also try other models available on OpenRouter (all use OpenAI SDK):

```python
# Anthropic Claude 3.5 Sonnet (paid)
citation_gen = CitationGenerator(
    model="anthropic/claude-3.5-sonnet",
    provider="openrouter"
)

# Meta Llama 3.1 70B (free)
citation_gen = CitationGenerator(
    model="meta-llama/llama-3.1-70b-instruct:free",
    provider="openrouter"
)

# Google Gemini Pro (via OpenRouter)
citation_gen = CitationGenerator(
    model="google/gemini-pro-1.5",
    provider="openrouter"
)

# OpenAI GPT-4 (via OpenRouter)
citation_gen = CitationGenerator(
    model="openai/gpt-4-turbo",
    provider="openrouter"
)
```

**Browse all models:** https://openrouter.ai/models

## Alternative: Google Gemini

### Why Gemini?
- ✅ **Fast responses** (~1-2s)
- ✅ **FREE tier** with generous limits
- ✅ **Good general knowledge**
- ✅ **Direct Google integration**

### Setup

1. Get API key from: https://makersuite.google.com/app/apikey
2. Add to `.env`:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```
3. In `src/api.py`, use:
   ```python
   citation_gen = CitationGenerator(
       model="gemini-2.0-flash-exp",
       provider="gemini"
   )
   ```

## Programmatic Usage

You can also initialize providers programmatically:

```python
from src.generation import CitationGenerator

# Option 1: OpenRouter (default)
generator = CitationGenerator()  # Uses OPENROUTER_API_KEY from env

# Option 2: Specify provider
generator = CitationGenerator(
    model="deepseek/deepseek-r1-0528:free",
    provider="openrouter"
)

# Option 3: Pass API key directly
generator = CitationGenerator(
    api_key="sk-or-v1-...",
    model="deepseek/deepseek-r1-0528:free",
    provider="openrouter"
)

# Option 4: Use Gemini
generator = CitationGenerator(
    model="gemini-2.0-flash-exp",
    provider="gemini"
)
```

## Performance Comparison

| Metric | DeepSeek R1 | Gemini 2.0 Flash |
|--------|-------------|------------------|
| **Speed** | ~2-3s | ~1-2s |
| **Cost** | FREE tier | FREE tier |
| **Reasoning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Context Following** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Medical Domain** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hallucination Control** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Setup Complexity** | Simple | Very Simple |

## Anti-Hallucination Features

Both providers benefit from:
- **Low temperature** (0.1) - deterministic responses
- **Strict prompting** - explicit instructions to avoid extrapolation
- **Source cleaning** - removes leaked citations
- **Confidence scoring** - tracks uncertainty

## Troubleshooting

### OpenRouter Errors

**401 Unauthorized**
- Check your API key is correct
- Verify it's set in `.env` as `OPENROUTER_API_KEY`

**429 Rate Limited**
- You've hit the FREE tier limit
- Wait a few minutes or upgrade to paid tier

**Model not found**
- Check model name is correct
- See available models at: https://openrouter.ai/models

### Gemini Errors

**Invalid API Key**
- Verify key from https://makersuite.google.com/app/apikey
- Check it's set as `GEMINI_API_KEY` in `.env`

**Safety Blocking**
- Our config disables most blocking
- Some medical content might still be blocked
- Try rephrasing or use OpenRouter

## Advanced Configuration

### Custom Parameters

```python
# For OpenRouter
citation_gen = CitationGenerator(
    model="deepseek/deepseek-r1-0528:free",
    provider="openrouter"
)
# Modify in generation.py _call_openrouter() for:
# - temperature (default: 0.1)
# - top_p (default: 0.8)
# - max_tokens (default: 1024)

# For Gemini
citation_gen = CitationGenerator(
    model="gemini-2.0-flash-exp",
    provider="gemini"
)
# Modify in __init__() GenerationConfig for:
# - temperature (default: 0.1)
# - top_p (default: 0.8)
# - top_k (default: 20)
# - max_output_tokens (default: 1024)
```

## Recommendations

**For Medical RAG Systems:**
👉 **Use DeepSeek R1** - Best balance of quality, cost, and hallucination control

**For Fast Prototyping:**
👉 **Use Gemini** - Quickest setup, very responsive

**For Production:**
👉 **Start with DeepSeek R1 FREE**, upgrade to paid OpenRouter credits if needed
