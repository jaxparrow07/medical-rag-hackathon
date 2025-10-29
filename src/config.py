# config.py - Comprehensive RAG System Configuration
import torch
from typing import Literal

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_CONFIG = {
    # Model Selection
    'model_name': 'pritamdeka/S-PubMedBert-MS-MARCO',  # Options: BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2, pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
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
    'provider': 'openrouter',  # Options: 'openrouter', 'gemini'
    
    # Model Selection by Provider
    'openrouter_model': 'minimax/minimax-m2:free',  # FREE options: z-ai/glm-4.5-air:free, deepseek/deepseek-r1, meta-llama/llama-3.2-3b-instruct:free
    'gemini_model': 'gemini-2.0-flash',  # Options: gemini-2.0-flash-exp, gemini-1.5-pro
    
    # Generation Parameters (for accuracy vs creativity
    'temperature': 0.3,  # Increased for more creative reasoning
    'top_p': 0.85,  # Increased for more varied responses
    'top_k': 40,  # Increased token sampling
    'max_output_tokens': 1000,  # Ensure complete explanations

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

'concise_medical_prompt': """You are a medical education assistant providing clear, brief explanations for medical questions.

## Available Context

{context}

## Instructions

- Provide concise, direct answers without excessive formatting
- For multiple-choice questions: State the answer, explain why it's correct in 2-3 sentences, then briefly note why other options don't fit
- Use the context when relevant, but also apply standard medical knowledge
- Write in short paragraphs with natural flow - no bullet points, no bold headers, no complex structure
- Get to the point quickly - students need clarity, not lengthy explanations
- Complete your explanation fully but keep it brief

## Question

{query}

## Your Answer

Provide a short, clear explanation:""",

'concise_medical_prompt2': """You are a medical education assistant providing clear, accurate explanations for medical questions and clinical scenarios.

## Available Context

{context}

## Instructions

When answering, think through the problem systematically:

**For straightforward questions:** State the answer directly and explain why in 2-3 sentences. Briefly note why other options are incorrect.

**For complex clinical scenarios:** Work through the case step-by-step:
1. Identify the key clinical features (age, symptoms, labs, timeline, risk factors)
2. Consider what diagnosis or management these findings suggest
3. Explain your reasoning by connecting the findings to the underlying pathophysiology or clinical principle
4. State your answer and why it's most appropriate for this specific patient

Use the provided context when relevant, but integrate it with standard medical knowledge and clinical reasoning. Don't limit yourself to only what's in the context - apply clinical logic and pattern recognition as physicians do.

**Style guidelines:**
- Write in natural flowing paragraphs, no bullet points or bold formatting
- Be concise but complete - explain your thinking without unnecessary detail
- Use clinical reasoning phrases like "This suggests...", "The combination of... points to...", "Given the presentation..."
- For time-sensitive or critical findings, note their clinical significance
- If findings are ambiguous, acknowledge this and explain which diagnosis is most likely based on the available information

**Critical thinking:**
- Always consider the timeline (acute vs chronic, progression)
- Weight findings by clinical significance (not all abnormalities are equally important)
- Think about mechanisms - why would this cause that?
- For treatment questions, consider the specific patient context (age, comorbidities, contraindications)

## Question

{query}

## Your Answer

Provide a clear, reasoned explanation:""",
    
    # Active prompt (selected by key)
    'active': 'concise_medical_prompt2'  # Options: 'main_prompt', 'concise_prompt', 'detailed_prompt'
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
RETRIEVAL_CONFIG = {
    'top_k': 6,  # Increased to get more context
    'similarity_threshold': 0.40,  # Lowered - medical terms might not match exactly
    'rerank': True,  # Keep reranking for accuracy
    'diversity_penalty': 0.2,  # Moderate diversity to avoid redundancy
    'context_window': 1500,  # Tokens per retrieved chunk
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
            'temperature': 0.35,
            'top_p': 0.8,
            'top_k': 35,
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
