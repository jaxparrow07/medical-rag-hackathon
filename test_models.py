"""
Test script for DeepSeek R1 integration with OpenRouter
"""
import os
from dotenv import load_dotenv
from src.generation import CitationGenerator

# Load environment variables
load_dotenv()

def test_deepseek_generation():
    """Test DeepSeek R1 with sample medical contexts"""
    
    # Initialize with OpenRouter + DeepSeek
    generator = CitationGenerator(
        model="deepseek/deepseek-r1",
        provider="openrouter"
    )
    
    # Sample medical contexts
    contexts = [
        {
            'text': 'Hypertension, also known as high blood pressure, is a condition where blood pressure in the arteries is persistently elevated. Normal blood pressure is typically around 120/80 mmHg. Hypertension is diagnosed when readings consistently exceed 140/90 mmHg.',
            'metadata': {
                'citation': 'Medical Guide - Cardiovascular Health, Page 42',
                'source': 'cardio_health.pdf'
            }
        },
        {
            'text': 'Treatment for hypertension often includes lifestyle modifications such as reducing salt intake, regular exercise, maintaining healthy weight, and limiting alcohol consumption. In many cases, medication may also be prescribed.',
            'metadata': {
                'citation': 'Hypertension Management Guidelines 2024, Page 15',
                'source': 'hypertension_guide.pdf'
            }
        }
    ]
    
    # Test query
    query = "What is hypertension and how is it treated?"
    
    print("🔬 Testing DeepSeek R1 Integration\n")
    print("=" * 70)
    print(f"Query: {query}\n")
    
    # Generate answer
    result = generator.generate_answer(query, contexts)
    
    print(f"Answer:\n{result['answer']}\n")
    print(f"Number of contexts used: {len(result['contexts'])}\n")
    print("=" * 70)

def test_gemini_comparison():
    """Compare with Gemini for reference"""
    
    try:
        generator = CitationGenerator(
            model="gemini-2.0-flash-exp",
            provider="gemini"
        )
        
        contexts = [
            {
                'text': 'Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels over a prolonged period. Type 2 diabetes is the most common form, accounting for about 90% of cases.',
                'metadata': {
                    'citation': 'Diabetes Handbook, Page 8',
                    'source': 'diabetes.pdf'
                }
            }
        ]
        
        query = "What is Type 2 diabetes?"
        
        print("\n🧪 Testing Gemini for Comparison\n")
        print("=" * 70)
        print(f"Query: {query}\n")
        
        result = generator.generate_answer(query, contexts)
        
        print(f"Answer:\n{result['answer']}\n")
        print(f"Confidence: {result['confidence']:.2f}")
        print("=" * 70)
        
    except ValueError as e:
        print(f"\n⚠️  Gemini test skipped: {e}")

if __name__ == "__main__":
    # Test DeepSeek
    test_deepseek_generation()
    
    # Optionally test Gemini
    test_gemini_comparison()
