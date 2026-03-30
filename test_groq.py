import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_groq_connectivity():
    api_key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用目前可用的最新模型
    data = {
        "model": "llama-3.3-70b-versatile", 
        "messages": [{"role": "user", "content": "Confirm if you are Llama 3.3 and working."}]
    }
    
    print(f"Testing Groq API with model: {data['model']}...")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print("\n[SUCCESS] Groq is alive!")
            print(f"Response: {answer}")
        else:
            print(f"\n[FAILED] Status Code: {response.status_code}")
            print(f"Error Detail: {response.text}")
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    test_groq_connectivity()