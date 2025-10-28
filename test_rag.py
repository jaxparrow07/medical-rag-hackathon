import requests
import json

def test_queries():
    queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes pneumonia?",  # With typo: "pnemonia"
    ]
    
    url = "http://localhost:8000/query"
    
    for query in queries:
        response = requests.post(url, json={"query": query, "top_k": 5})
        result = response.json()
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Contexts: {len(result['contexts'])} retrieved")

if __name__ == "__main__":
    test_queries()

