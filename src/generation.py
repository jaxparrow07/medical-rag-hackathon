import google.generativeai as genai
from typing import List, Dict

class CitationGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize with Gemini 2.5 Flash (using 2.0-flash-exp as latest available)"""
        genai.configure(api_key=api_key)
        
        # Configure for more deterministic outputs
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Lower temperature for more factual responses
            top_p=0.8,
            top_k=20,
            max_output_tokens=1024,
        )
        
        self.model = genai.GenerativeModel(
            model,
            generation_config=generation_config
        )
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> Dict:
        """Generate answer with citations"""
        
        # Format contexts with citations
        context_text = self.format_contexts(contexts)
        
        prompt = f"""You are a medical information assistant. Your task is to provide clear, accurate answers based STRICTLY on the provided medical context.

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
{context_text}

**Question:** {query}

**Your Answer (clear, accessible, fact-based only):**"""

        try:
            response = self.model.generate_content(
                prompt,
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE_SPEECH': 'block_none',
                    'SEXUALLY_EXPLICIT': 'block_none',
                    'DANGEROUS_CONTENT': 'block_none'
                }
            )
            
            answer = response.text
            
            # Post-process to ensure no source citations leaked through
            answer = self._clean_source_mentions(answer)
            
            return {
                'answer': answer,
                'contexts': [ctx['text'] for ctx in contexts],
                'citations': [ctx['metadata']['citation'] for ctx in contexts],
                'confidence': self._assess_confidence(answer, contexts)
            }
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error generating the response: {str(e)}",
                'contexts': [],
                'citations': [],
                'confidence': 0.0
            }
    
    def format_contexts(self, contexts: List[Dict]) -> str:
        """Format contexts without exposing source numbering"""
        formatted = []
        for ctx in contexts:
            text = ctx['text'].strip()
            formatted.append(text)
        
        return "\n\n".join(formatted)
    
    def _clean_source_mentions(self, text: str) -> str:
        """Remove any source citations that leaked into the answer"""
        import re
        # Remove patterns like [Source X], (Source X), etc.
        patterns = [
            r'\[Source \d+[^\]]*\]',
            r'\(Source \d+[^\)]*\)',
            r'Source \d+:',
            r'According to Source \d+',
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _assess_confidence(self, answer: str, contexts: List[Dict]) -> float:
        """Assess confidence based on answer characteristics"""
        uncertainty_phrases = [
            'does not contain',
            'insufficient',
            'unclear',
            'not specified',
            'not enough information',
            'cannot determine',
            'not mentioned'
        ]
        
        answer_lower = answer.lower()
        
        # Low confidence if uncertainty phrases are present
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return 0.3
        
        # Medium-high confidence otherwise
        return 0.8
