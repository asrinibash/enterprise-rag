import sys
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/api/v1"

# Test queries
TEST_QUERIES = [
    "What is the main topic of the documents?",
    "Explain the key concepts discussed",
    "What are the most important points?",
]


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 80)
    print("Testing Health Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"LLM Available: {data['llm_available']}")
        print(f"Index Stats: {data['index_stats']}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\n" + "=" * 80)
    print("System Statistics")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/stats")
        response.raise_for_status()
        
        data = response.json()
        print(json.dumps(data, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ Stats request failed: {e}")
        return False


def test_query(query: str, top_k: int = 3):
    """Test a single query."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_citations": True
        }
        
        response = requests.post(f"{API_URL}/query", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"\n📊 Performance:")
        print(f"  Retrieval: {data['retrieval_time_ms']:.2f}ms")
        print(f"  Generation: {data['generation_time_ms']:.2f}ms")
        print(f"  Total: {data['total_time_ms']:.2f}ms")
        
        print(f"\n💬 Answer ({data['model_used']}):")
        print(f"  {data['answer']}")
        
        print(f"\n📚 Sources ({len(data['sources'])}):")
        for i, source in enumerate(data['sources'], 1):
            print(f"  [{i}] {source['metadata']['file_name']}")
            print(f"      {source['content'][:150]}...")
        
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("❌ No relevant documents found")
        else:
            print(f"❌ Query failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False


def test_list_documents():
    """Test document listing."""
    print("\n" + "=" * 80)
    print("Indexed Documents")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_URL}/documents")
        response.raise_for_status()
        
        data = response.json()
        print(f"Total documents: {data['total_documents']}\n")
        
        for doc in data['documents']:
            print(f"📄 {doc['file_name']}")
            print(f"   Type: {doc['file_type']}, Chunks: {doc['chunks']}")
        
        return True
    except Exception as e:
        print(f"❌ Document listing failed: {e}")
        return False


def interactive_mode():
    """Interactive query mode."""
    print("\n" + "=" * 80)
    print("Interactive Query Mode")
    print("=" * 80)
    print("Enter your queries (type 'quit' to exit)\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            test_query(query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main test function."""
    print("\n🚀 RAG System Test Suite")
    
    # Check if API is running
    if not test_health():
        print("\n❌ API is not running!")
        print("Start the API with: uv run python -m src.main")
        return
    
    # Run tests
    test_stats()
    test_list_documents()
    
    # Test predefined queries
    print("\n" + "=" * 80)
    print("Testing Predefined Queries")
    print("=" * 80)
    
    for query in TEST_QUERIES:
        test_query(query)
    
    # Interactive mode
    print("\n" + "=" * 80)
    choice = input("Enter interactive mode? (y/n): ").strip().lower()
    if choice == 'y':
        interactive_mode()


if __name__ == "__main__":
    main()