import requests

def query_ollama(prompt):
    url = "http://localhost:11411/api/llama3"  # Adjust the URL if needed
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json().get('completion', '')
    else:
        raise Exception(f"Error querying Ollama: {response.status_code}, {response.text}")
