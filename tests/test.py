import requests
import json

url = "http://localhost:8000/predict"

with open('test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

response = requests.post(url, json=test)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())