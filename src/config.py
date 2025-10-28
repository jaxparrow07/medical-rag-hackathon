# config.py - Comprehensive RAG System Configuration
import torch
from typing import Literal

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_CONFIG = {
    # Model Selection
    'model_name': 'BAAI/bge-base-en-v1.5',  # Options: BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2, pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
    'max_seq_length': 512,
    
    # Precision Settings (for accuracy vs speed trade-off)
    'dtype': 'float32',  # Options: 'float32' (accurate), 'float16' (fast)
    'use_mixed_precision': False,  # True for speed, False for accuracy
    
    # Batch Processing
    'batch_size': 16,  # Adjust based on VRAM: 16 (accurate), 32 (balanced), 64 (fast)
    
    # GPU Optimizations
    'enable_cudnn_benchmark': False,  # True for speed, False for deterministic results
    'deterministic': True,  # True for reproducibility, False for speed
    'allow_tf32': False,  # True for RTX 30/40 series speed, False for accuracy
    'matmul_precision': 'highest',  # Options: 'highest' (accurate), 'high' (balanced), 'medium' (fast)
    
    # Normalization
    'normalize_embeddings': True,  # Always True for cosine similarity
    'add_query_prefix': True,  # Improves BGE model performance
}

# ============================================================================
# LLM GENERATION CONFIGURATION
# ============================================================================
LLM_CONFIG = {
    # Provider Selection
    'provider': 'gemini',  # Options: 'openrouter', 'gemini'
    
    # Model Selection by Provider
    'openrouter_model': 'tngtech/deepseek-r1t2-chimera:free',  # FREE options: z-ai/glm-4.5-air:free, deepseek/deepseek-r1, meta-llama/llama-3.2-3b-instruct:free
    'gemini_model': 'gemini-2.0-flash',  # Options: gemini-2.0-flash-exp, gemini-1.5-pro
    
    # Generation Parameters (for accuracy vs creativity)
    'temperature': 0.15,  # 0.0-0.3 (factual), 0.4-0.7 (balanced), 0.8-1.0 (creative)
    'top_p': 0.8,  # 0.8-0.9 (focused), 0.9-0.95 (balanced), 0.95-1.0 (diverse)
    'top_k': 20,  # 10-20 (focused), 20-40 (balanced), 40-100 (diverse)
    'max_output_tokens': 2048,  # Max tokens in response
    
    # Response Quality
    'enable_source_cleaning': True,  # Remove leaked source citations
    'assess_confidence': True,  # Calculate answer confidence score
}

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================
SYSTEM_PROMPTS = {
    # Main prompt template
    'main_prompt': """You are a medical information assistant. Your task is to provide clear, accurate answers based STRICTLY on the provided medical context.

**CRITICAL RULES:**
1. Answer ONLY using information explicitly stated in the context below
2. DO NOT mention "Source 1", "Source 2", etc. in your response
3. DO NOT add any information, assumptions, or inferences beyond what's in the context
4. DO NOT extrapolate, generalize, or make logical leaps
5. If information is insufficient or unclear, explicitly state: "The provided information does not contain enough details to answer this fully."
6. Use simple, clear language that a non-medical person can understand
7. Avoid medical jargon where possible; if technical terms are necessary, briefly explain them
8. Structure your answer in short, clear sentences

**Medical Context:**
{context}

**Question:** {query}

**Your Answer (clear, accessible, fact-based only):**""",
    
    # Alternative: More concise prompt
    'concise_prompt': """Based ONLY on the medical context below, answer the question clearly and simply.

Rules:
- Use ONLY the provided information
- No source citations in your answer
- Simple language for non-medical readers
- If unclear, say "Insufficient information"

Context:
{context}

Question: {query}

Answer:""",
    
    # Alternative: Detailed medical prompt
    'detailed_prompt': """You are a medical information specialist. Provide a comprehensive answer based strictly on the context.

Guidelines:
1. Ground every statement in the provided context
2. Do not reference sources by number (Source 1, etc.)
3. Use medical terminology with explanations
4. Structure: Overview → Details → Summary
5. Acknowledge information gaps explicitly

Medical Context:
{context}

Patient Question: {query}

Detailed Response:""",

'natural_medical_prompt': """You are a trusted medical information assistant helping people understand health information clearly. Your role is to explain medical concepts in natural, conversational language while staying completely faithful to the provided context.

Core Instructions:

Answer using only the information explicitly stated in the medical context below. Write in complete paragraphs as if you're explaining something to a friend or family member, using natural flowing sentences rather than bullet points or numbered lists. When the context mentions side effects, medications, dosages, or warnings, include these details clearly in your explanation without downplaying their importance.

Use everyday language whenever possible. If you must use a medical term, briefly explain what it means in plain English within the same sentence. Never reference sources by number or mention where information came from. Simply present the information as unified knowledge.

If the context discusses potential risks, side effects, or precautions, weave these naturally into your explanation using phrases like "it's important to note that" or "patients should be aware that" rather than creating separate warning sections. When medications are mentioned, include relevant details about how they work, when they're used, and any important considerations the context provides.

Structure your response with simple headings only when the answer covers multiple distinct topics that would genuinely benefit from separation. Otherwise, write in flowing paragraphs that read naturally from start to finish.

Use line breaks whenever needed to separate text blocks.

CRITICAL: Always talk in a professional manner to convey information in a simple way. Do not ever mentioned that you have been given information or data. TALK CASUALLY. You must complete your entire response. Never stop mid-sentence or mid-explanation. If the answer requires multiple paragraphs, write all of them fully. Ensure every thought is finished, every explanation is complete, and the response ends naturally at a proper conclusion. Do not truncate or leave any information incomplete.

Medical Context:
{context}

Question: {query}

Your Response:""",
    
    # Active prompt (selected by key)
    'active': 'natural_medical_prompt'  # Options: 'main_prompt', 'concise_prompt', 'detailed_prompt'
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
RETRIEVAL_CONFIG = {
    'top_k': 3,  # Number of contexts to retrieve (3-10)
    'similarity_threshold': 0.6,  # Minimum similarity score (0.0-1.0)
    'rerank': True,  # Enable cross-encoder reranking (slower but more accurate)
    'diversity_penalty': 0.0,  # 0.0 (no penalty) to 0.5 (diverse results)
}

# ============================================================================
# PERFORMANCE PRESETS
# ============================================================================
PRESETS = {
    'maximum_accuracy': {
        'embedding': {
            'dtype': 'float32',
            'use_mixed_precision': False,
            'batch_size': 16,
            'enable_cudnn_benchmark': False,
            'deterministic': True,
            'allow_tf32': False,
            'matmul_precision': 'highest',
        },
        'llm': {
            'temperature': 0.05,
            'top_p': 0.75,
            'top_k': 10,
        }
    },
    
    'balanced': {
        'embedding': {
            'dtype': 'float32',
            'use_mixed_precision': False,
            'batch_size': 32,
            'enable_cudnn_benchmark': False,
            'deterministic': True,
            'allow_tf32': False,
            'matmul_precision': 'high',
        },
        'llm': {
            'temperature': 0.2,
            'top_p': 0.8,
            'top_k': 20,
        }
    },
    
    'maximum_speed': {
        'embedding': {
            'dtype': 'float16',
            'use_mixed_precision': True,
            'batch_size': 64,
            'enable_cudnn_benchmark': True,
            'deterministic': False,
            'allow_tf32': True,
            'matmul_precision': 'medium',
        },
        'llm': {
            'temperature': 0.1,
            'top_p': 0.8,
            'top_k': 20,
        }
    }
}

# ============================================================================
# ACTIVE PRESET
# ============================================================================
ACTIVE_PRESET = 'balanced'  # Options: 'maximum_accuracy', 'balanced', 'maximum_speed'

# Apply preset to configs
if ACTIVE_PRESET in PRESETS:
    EMBEDDING_CONFIG.update(PRESETS[ACTIVE_PRESET]['embedding'])
    LLM_CONFIG.update(PRESETS[ACTIVE_PRESET]['llm'])

# ============================================================================
# BENCHMARKING MODE
# ============================================================================
BENCHMARK_CONFIG = {
    'enabled': False,  # Set to True to test different configurations
    'test_configs': [
        {'name': 'Accurate', 'preset': 'maximum_accuracy'},
        {'name': 'Balanced', 'preset': 'balanced'},
        {'name': 'Fast', 'preset': 'maximum_speed'},
    ],
    'test_queries': [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes migraine headaches?",
    ],
    'metrics': ['latency', 'similarity_score', 'confidence'],
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
HARDWARE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gpu_name': 'RTX 4050',  # For documentation
    'vram_gb': 6,
    'ram_gb': 16,
    'cpu': 'Ryzen 7 7000',
}

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================
DEBUG_CONFIG = {
    'verbose': True,
    'log_embeddings': False,
    'log_retrievals': True,
    'log_llm_calls': True,
    'save_failed_queries': True,
}
