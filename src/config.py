# config.py - Comprehensive RAG System Configuration
import torch
from typing import Literal

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
EMBEDDING_CONFIG = {
    # Model Selection - Medical single-encoder (works with query reformulation)
    # BioLORD: Medical model trained on UMLS, handles medical synonyms and concepts
    'model_name': 'FremyCompany/BioLORD-2023-M',  # Medical single encoder
    'article_encoder': None,  # None = single encoder mode

    'max_seq_length': 512,  # Standard BERT length
    'embedding_dim': 768,  # BERT base dimension

    # Precision Settings (for accuracy vs speed trade-off)
    'dtype': 'float32',  # Options: 'float32' (accurate), 'float16' (fast)
    'use_mixed_precision': False,  # True for speed, False for accuracy

    # Batch Processing
    'batch_size': 32,  # Adjust based on VRAM: 16 (accurate), 32 (balanced), 64 (fast)

    # GPU Optimizations
    'enable_cudnn_benchmark': False,  # True for speed, False for deterministic results
    'deterministic': True,  # True for reproducibility, False for speed
    'allow_tf32': False,  # True for RTX 30/40 series speed, False for accuracy
    'matmul_precision': 'highest',  # Options: 'highest' (accurate), 'high' (balanced), 'medium' (fast)

    # Normalization
    'normalize_embeddings': True,  # Always True for cosine similarity
    'add_query_prefix': False,  # Improves BGE model performance (ignored for MedCPT)
}

# ============================================================================
# LLM GENERATION CONFIGURATION
# ============================================================================
LLM_CONFIG = {
    # Provider Selection
    'provider': 'openrouter',  # Options: 'openrouter', 'gemini'

    # Model Selection by Provider (both via OpenRouter)
    'openrouter_model': 'minimax/minimax-m2',  # MiniMax M2 for answer generation
    'gemini_model': 'gemini-2.0-flash',  # Options: gemini-2.0-flash-exp, gemini-1.5-pro

    # Generation Parameters (for accuracy vs creativity)
    'temperature': 0.3,  # Increased for more creative reasoning
    'top_p': 0.85,  # Increased for more varied responses
    'top_k': 40,  # Increased token sampling
    'max_output_tokens': 1000,  # Ensure complete explanations

    # Response Quality
    'enable_source_cleaning': True,  # Remove leaked source citations
    'assess_confidence': True,  # Calculate answer confidence score
}

# ============================================================================
# QUERY REFORMULATION CONFIGURATION
# ============================================================================
QUERY_REFORMULATION_CONFIG = {
    'enabled': True,  # Enable query reformulation
    'model': 'minimax/minimax-01',  # MiniMax model via OpenRouter (compatible identifier)
    'num_variants': 3,  # Number of query variants to generate
    'temperature': 0.4,  # Balanced creativity and consistency
    'use_reasoning': True,  # Use reasoning-based reformulation (slower, more detailed)
    'fallback_to_original': True,  # Use original query if reformulation fails
}

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNKING_CONFIG = {
    'strategy': 'semantic',  # Options: 'word_based' (legacy), 'semantic' (recommended), 'heading_based'
    'max_tokens': EMBEDDING_CONFIG['max_seq_length'],  # Match embedding model max_seq_length
    'similarity_threshold': 0.75,  # For semantic chunking (0.0-1.0)
    'min_chunk_tokens': 100,  # Minimum tokens per chunk
    'preserve_sentences': True,  # Never split mid-sentence (semantic chunking only)
}

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
RETRIEVAL_CONFIG = {
    # Multi-Query Retrieval
    'strategy': 'multi_query',  # Options: 'single', 'multi_query', 'hybrid'
    'top_k_per_query': 10,  # Retrieve N results per query variant
    'final_top_k': 6,  # Final number of results after merging
    'top_k': 6,  # Default top_k for API compatibility

    # Similarity Filtering
    'similarity_threshold': 0.5,  # Minimum similarity score (0.0-1.0) - ensures relevant results only

    # Reranking (Cross-Encoder)
    'rerank': True,  # Enable cross-encoder reranking
    'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Cross-encoder model
    'rerank_top_k': 6,  # Top-k after reranking

    # Legacy
    'diversity_penalty': 0.2,  # Not implemented yet
    'context_window': 1500,  # Tokens per retrieved chunk (for display)
}

# ============================================================================
# HYBRID SEARCH CONFIGURATION
# ============================================================================
HYBRID_SEARCH_CONFIG = {
    'enabled': False,  # Enable hybrid search (BM25 + Vector)
    'alpha': 0.5,  # BM25 weight (0=pure vector, 1=pure BM25, 0.5=balanced)
    'adaptive': True,  # Auto-adjust alpha based on query type
    'use_for_query_types': ['abbreviation', 'drug_name', 'protocol'],  # When to use hybrid
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

'concise_medical_prompt2': """
You are a medical education assistant providing clear, accurate, exam‑style explanations.

Available Context:
{context}

Instructions:

    Produce a one‑best‑answer explanation in the following format:

        Answer: <Letter> — <Diagnosis/Action>.

        Key finding: 1 short clause that clinches the diagnosis/decision.

        Mechanism: 1 sentence linking the key finding to the principle/pathophysiology.

        Tie‑back: 1 sentence that explicitly links the mechanism to the chosen answer.

    Length: 2–4 sentences total; no lists; no extra teaching points.

    Tone: definitive, mechanism‑based, and concise; avoid hedging, qualifiers, or discussion of other options unless asked.

For basic science stems:

    Answer: <Concept>.

    Principle: 1–2 sentences stating the governing mechanism or pathway.

For clinical stems:

    Identify the single highest‑yield finding → state the mechanism → state the diagnosis/next step.

Context use:

    Use {context} only if it directly supports the mechanism; ignore irrelevant context.

    Do not quote or enumerate sources; no citations or references in the output.

Input:
{query}

Output (single paragraph):
Answer: <Letter> — <Final choice>. Key finding: <clincher>. Mechanism: <mechanism>. Therefore, <final link to the choice>.

Constraints:

    2–4 sentences; no discussion of other options; no restating the entire vignette.
r""",
    
    # Active prompt (selected by key)
    'active': 'concise_medical_prompt2'  # Options: 'main_prompt', 'concise_prompt', 'detailed_prompt'
}

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================
HARDWARE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ============================================================================
# ENHANCED FEATURES CONFIGURATION
# ============================================================================
ENHANCED_FEATURES_CONFIG = {
    # PDF Processing
    'extract_tables': False,  # Extract tables from PDFs using pdfplumber !!! REALLY SLOW !!!

    # Medical NER
    'use_medical_ner': True,  # Use scispacy for medical entity recognition
    'ner_model': 'en_core_sci_md',  # Options: en_core_sci_sm, en_core_sci_md, en_core_sci_lg

    # Query Classification
    'use_query_classification': True,  # Automatically route queries to optimal strategy
    'auto_adjust_retrieval': True,  # Adjust retrieval based on query type

    # Context Expansion
    'use_context_expansion': True,  # Include surrounding chunks
    'context_window_size': 1,  # Number of chunks before/after to include

    # Evaluation
    'enable_evaluation': True,  # Track retrieval metrics and performance
    'save_evaluation_reports': True,  # Save evaluation reports to disk
    'evaluation_dir': './data/evaluation',  # Directory for evaluation reports
}

# ============================================================================
# PDF EXTRACTION CONFIGURATION
# ============================================================================
PDF_EXTRACTION_CONFIG = {
    # Table Extraction
    'extract_tables': ENHANCED_FEATURES_CONFIG['extract_tables'],
    'format_tables_as_text': True,  # Convert tables to natural language

    # Structure Detection
    'detect_headers': True,  # Detect section headers
    'extract_acronyms': True,  # Extract acronym definitions

    # Quality Validation
    'validate_pdf_quality': True,  # Check if PDF needs OCR
    'min_chars_per_page': 300,  # Minimum chars for "medium" quality

    # Metadata Extraction
    'extract_pdf_metadata': True,  # Extract author, title, dates, etc.
}

# ============================================================================
# MEDICAL NER CONFIGURATION
# ============================================================================
MEDICAL_NER_CONFIG = {
    'enabled': ENHANCED_FEATURES_CONFIG['use_medical_ner'],
    'model_name': ENHANCED_FEATURES_CONFIG['ner_model'],

    # Entity Types to Extract
    'extract_diseases': True,
    'extract_drugs': True,
    'extract_procedures': True,
    'extract_anatomy': True,

    # Metadata Enrichment
    'enrich_chunk_metadata': True,  # Add medical entities to chunk metadata
    'extract_abbreviations': True,  # Extract medical abbreviations
}

# ============================================================================
# QUERY CLASSIFICATION CONFIGURATION
# ============================================================================
QUERY_CLASSIFICATION_CONFIG = {
    'enabled': ENHANCED_FEATURES_CONFIG['use_query_classification'],

    # Strategy Routing
    'auto_route_strategy': ENHANCED_FEATURES_CONFIG['auto_adjust_retrieval'],

    # Confidence Threshold
    'min_confidence_for_routing': 0.7,  # Minimum confidence to use recommended strategy

    # Fallback Strategy
    'fallback_strategy': 'multi_query',  # Default if classification uncertain
}

# ============================================================================
# CONTEXT EXPANSION CONFIGURATION
# ============================================================================
CONTEXT_EXPANSION_CONFIG = {
    'enabled': ENHANCED_FEATURES_CONFIG['use_context_expansion'],
    'window_size': ENHANCED_FEATURES_CONFIG['context_window_size'],

    # Context Merging
    'merge_with_main_chunk': True,  # Combine context with main chunk for LLM
    'include_context_markers': False,  # Add "CONTEXT BEFORE/AFTER" markers

    # Position Tracking
    'track_chunk_positions': True,  # Track chunk positions during ingestion
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    'enabled': ENHANCED_FEATURES_CONFIG['enable_evaluation'],
    'save_reports': ENHANCED_FEATURES_CONFIG['save_evaluation_reports'],
    'results_dir': ENHANCED_FEATURES_CONFIG['evaluation_dir'],

    # Metrics to Track
    'track_performance': True,  # Latency, throughput
    'track_feedback': True,  # User ratings and feedback
    'track_faithfulness': True,  # Answer faithfulness to context

    # Reporting
    'auto_save_interval': 100,  # Save report every N queries
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

    'log_query_classification': True,
    'log_medical_entities': False,
    'log_context_expansion': True,
    'log_evaluation_metrics': True,
}
