import requests

url = "http://localhost:8000/predict"

reviews = [
    "Absolutely loved this product, works perfectly",
    "Complete garbage, broke after one day",
    "It is fine, does what it says"
]

for review in reviews:
    response = requests.post(url, json={"text": review})
    print(response.json())