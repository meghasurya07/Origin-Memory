"""
Origin Brain — REST API Server Example

Shows how to run the API server and make requests.
Start with: python examples/05_rest_api.py
Then test with curl or use the interactive docs at http://localhost:8000/docs
"""
import sys
import os

def print_curl_examples():
    """Print curl commands for testing the API."""
    BASE = "http://localhost:8000"
    
    print("=== Origin Brain REST API ===")
    print(f"\nSwagger UI:  {BASE}/docs")
    print(f"ReDoc:       {BASE}/redoc")
    print(f"Health:      {BASE}/v1/health")
    
    print("\n--- Example curl commands ---\n")
    
    print("# 1. Health check")
    print(f"curl {BASE}/v1/health\n")
    
    print("# 2. Encode a memory")
    print(f"""curl -X POST {BASE}/v1/encode \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_id": "agent_1", "content": "User prefers Python and dark mode"}}'\n""")
    
    print("# 3. Recall memories")
    print(f"""curl -X POST {BASE}/v1/recall \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_id": "agent_1", "query": "user preferences", "top_k": 5}}'\n""")
    
    print("# 4. Sleep consolidation")
    print(f"""curl -X POST {BASE}/v1/sleep \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_id": "agent_1"}}'\n""")
    
    print("# 5. Publish to shared memory")
    print(f"""curl -X POST {BASE}/v1/shared/publish \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_id": "agent_1", "key": "team_goal", "value": "Ship v1.0 by December"}}'\n""")
    
    print("# 6. Query shared memory")
    print(f"""curl -X POST {BASE}/v1/shared/query \\
  -H "Content-Type: application/json" \\
  -d '{{"agent_id": "agent_2", "key": "team_goal"}}'\n""")
    
    print("# 7. Get brain stats")
    print(f"curl {BASE}/v1/stats/agent_1\n")
    
    print("# 8. List all agents")
    print(f"curl {BASE}/v1/agents\n")


if __name__ == "__main__":
    if "--run" in sys.argv:
        # Actually start the server
        try:
            import uvicorn
            print("Starting Origin Brain API server on http://localhost:8000")
            print("Press Ctrl+C to stop\n")
            uvicorn.run("origin_brain.server:app", host="0.0.0.0", port=8000, reload=True)
        except ImportError:
            print("Error: uvicorn not installed. Run: pip install origin-brain[server]")
            sys.exit(1)
    else:
        print_curl_examples()
        print("\n--- To start the server: python examples/05_rest_api.py --run ---")
