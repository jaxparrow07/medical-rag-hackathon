"""
Quick test to verify the frontend server is working
"""

import requests
import json

def test_frontend_server():
    """Test the frontend server endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Medical RAG Frontend Server")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Documents: {data.get('documents')}")
            print(f"   Provider: {data.get('config', {}).get('provider')}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print(f"   Make sure the server is running: python server.py")
        return
    
    # Test 2: Stats endpoint
    print("\n2. Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Stats retrieved")
            print(f"   Total documents: {data.get('total_documents')}")
            print(f"   LLM Model: {data.get('llm_config', {}).get('model')}")
        else:
            print(f"   ✗ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Query endpoint
    print("\n3. Testing query endpoint...")
    try:
        test_query = "What are the symptoms of diabetes?"
        print(f"   Query: {test_query}")
        
        response = requests.post(
            f"{base_url}/api/query",
            json={"query": test_query},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Query successful")
            print(f"   Answer preview: {data.get('answer', '')[:100]}...")
            print(f"   Contexts retrieved: {len(data.get('context', []))}")
        else:
            print(f"   ✗ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("\nIf all tests passed, open http://localhost:8000 in your browser!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_frontend_server()
