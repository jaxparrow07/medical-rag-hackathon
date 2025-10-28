import google.generativeai as genai
from typing import List, Dict

class CitationGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize with Gemini 2.5 Flash (using 2.0-flash-exp as latest available)"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> Dict:
        """Generate answer with citations"""
        
        # Format contexts with citations
        context_text = self.format_contexts(contexts)
        
        prompt = f"""You are a medical expert assistant. Answer the question using ONLY the provided context.

**IMPORTANT RULES:**
1. Base your answer ONLY on the provided context
2. After each claim, cite the source using [Source X, Page Y] format
3. If the context doesn't contain enough information, say so clearly
4. Do not add information not present in the context
5. Be precise and medical in your language

**Context:**
{context_text}

**Question:** {query}

**Answer with citations:**"""

        response = self.model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'contexts': [ctx['text'] for ctx in contexts],
            'citations': [ctx['metadata']['citation'] for ctx in contexts]
        }
    
    def format_contexts(self, contexts: List[Dict]) -> str:
        """Format contexts with source information"""
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx['metadata']['citation']
            text = ctx['text']
            formatted.append(f"[Source {i} - {source}]\n{text}\n")
        
        return "\n".join(formatted)
