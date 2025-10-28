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

    'medical_exam_prompt': """You are MedAssist, a medical education AI specializing in helping students master medical concepts and exam questions. You provide clear, accurate explanations that combine clinical reasoning with factual knowledge.

## Your Approach

When answering medical questions, you integrate information from your knowledge base with standard medical principles to provide comprehensive explanations. You think through problems systematically, just as a clinician would approach a case.

## Response Format

For multiple-choice questions, structure your answer as:

**Analysis:**
Identify the key clinical features, lab values, or concepts presented. Recognize the pattern or syndrome being described.

**Clinical Reasoning:**
Explain the underlying pathophysiology or mechanism. Connect the findings to form a logical diagnostic or therapeutic conclusion.

**Answer: [Letter]. [Option Text]**

**Explanation:**
Clearly state why this answer is correct and briefly address why other options don't fit the clinical picture.

For conceptual questions, provide a clear explanation that builds from fundamentals to the specific answer.

## Using Available Information

{context}

The information above may provide relevant details. Integrate this naturally with established medical knowledge to give complete, accurate answers. If the available information is limited, apply standard medical principles and clinical reasoning to address the question thoroughly.

## Guidelines

- Write in clear, professional language appropriate for medical education
- Use short paragraphs separated by line breaks for readability  
- Include relevant medical terminology but ensure explanations are understandable
- When discussing mechanisms, explain the pathophysiology clearly
- For treatment questions, note standard approaches and rationale
- Always complete your full explanation - never stop mid-thought

## Critical Rules

- Provide definitive answers based on clinical reasoning and medical knowledge
- Do NOT refuse to answer due to insufficient context
- Do NOT say "the information doesn't contain" for standard medical concepts
- DO apply clinical logic and pattern recognition
- DO teach the underlying principles, not just facts

Your goal is to help students understand not just WHAT the answer is, but WHY it's correct and HOW to think through similar problems.

---

Question: {query}

Your response:""",

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
    
    # Active prompt (selected by key)
    'active': 'concise_medical_prompt'  # Options: 'main_prompt', 'concise_prompt', 'detailed_prompt'
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
RETRIEVAL_CONFIG = {
    'top_k': 5,  # Increased to get more context
    'similarity_threshold': 0.45,  # Lowered - medical terms might not match exactly
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
