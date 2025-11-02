import google.generativeai as genai
from typing import List, Dict, Literal
import os
import re
from openai import OpenAI

try:
    from .config import LLM_CONFIG, SYSTEM_PROMPTS, DEBUG_CONFIG
except ImportError:
    from src.config import LLM_CONFIG, SYSTEM_PROMPTS, DEBUG_CONFIG

class CitationGenerator:
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        provider: Literal["openrouter", "gemini"] = None
    ):
        """
        Initialize with config-driven settings

        Args:
            api_key: API key for the provider (uses env vars if None)
            model: Model identifier (uses config default if None)
            provider: "openrouter" or "gemini" (uses config default if None)
        """
        # Use config defaults if not provided
        self.provider = provider or LLM_CONFIG['provider']

        if self.provider == "openrouter":
            self.model_name = model or LLM_CONFIG['openrouter_model']
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment or parameters")

            # Initialize OpenAI client pointing to OpenRouter
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/jaxparrow07/rag-model",
                    "X-Title": "Medical RAG System"
                }
            )

        elif self.provider == "gemini":
            self.model_name = model or LLM_CONFIG['gemini_model']
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or parameters")
            genai.configure(api_key=self.api_key)

            # Configure for more deterministic outputs using config
            generation_config = genai.GenerationConfig(
                temperature=LLM_CONFIG['temperature'],
                top_p=LLM_CONFIG['top_p'],
                top_k=LLM_CONFIG['top_k'],
                max_output_tokens=LLM_CONFIG['max_output_tokens'],
            )

            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config=generation_config
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if DEBUG_CONFIG['verbose']:
            print(f"Initialized {self.provider} with model: {self.model_name}")
            print(f"Temperature: {LLM_CONFIG['temperature']}, Top-p: {LLM_CONFIG['top_p']}")
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> Dict:
        """Generate answer with config-driven prompt"""
        context_text = self.format_contexts(contexts)
        
        # Get active prompt template from config
        prompt_template = SYSTEM_PROMPTS[SYSTEM_PROMPTS['active']]
        prompt = prompt_template.format(context=context_text, query=query)
        
        if DEBUG_CONFIG['log_llm_calls']:
            print(f"Generating answer for: {query[:100]}...")

        try:
            if self.provider == "openrouter":
                answer = self._call_openrouter(prompt)
            else:  # gemini
                answer = self._call_gemini(prompt)
            
            # Post-process if enabled in config
            if LLM_CONFIG['enable_source_cleaning']:
                answer = self._clean_source_mentions(answer)
            
            result = {
                'answer': answer,
                'contexts': [ctx['text'] for ctx in contexts]
            }
            
            # Add confidence if enabled in config
            if LLM_CONFIG['assess_confidence']:
                result['confidence'] = self._assess_confidence(answer, contexts)

            if DEBUG_CONFIG['log_llm_calls']:
                print("-" * 50)
                print(f"Prompt used: {prompt[:80]}...\n")
            
            return result
            
        except Exception as e:
            if DEBUG_CONFIG['verbose']:
                print(f"Error generating answer: {e}")
            return {
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'contexts': []
            }
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API with config settings"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG['temperature'],
                top_p=LLM_CONFIG['top_p'],
                max_tokens=LLM_CONFIG['max_output_tokens']
            )

            # Get the message content
            message = response.choices[0].message
            content = message.content

            # Debug: Log the type of content if verbose
            if DEBUG_CONFIG.get('verbose', False):
                print(f"DEBUG: Response type: {type(content)}")
                if hasattr(message, '__dict__'):
                    print(f"DEBUG: Message attributes: {message.__dict__}")

            # Handle different response formats
            if isinstance(content, dict):
                # DeepSeek R1 may return structured content
                return content.get('content', content.get('text', str(content)))
            elif content is None:
                # Handle None responses
                if DEBUG_CONFIG.get('verbose', False):
                    print(f"DEBUG: Full response: {response}")
                return ""
            else:
                # Standard string response
                return str(content)

        except Exception as e:
            if DEBUG_CONFIG.get('verbose', False):
                import traceback
                print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with config settings"""
        response = self.model.generate_content(
            prompt,
            safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            }
            ],
        )
        return response.text
    
    def format_contexts(self, contexts: List[Dict]) -> str:
        """Format contexts without exposing source numbering"""
        formatted = []
        for ctx in contexts:
            text = ctx['text'].strip()
            formatted.append(text)
        
        return "\n\n".join(formatted)
    
    def _clean_source_mentions(self, text: str) -> str:
        """Remove source citations from answer"""
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
        """Assess confidence in answer"""
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
        
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return 0.3
        
        return 0.8
