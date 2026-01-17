import requests
import sys

BASE_URL = "http://127.0.0.1:8001"
IMAGE_PATH = "biryani.jpg"

def test_root():
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint Check Passed:", response.json())
        else:
            print(f"❌ Root endpoint Check Failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")

def test_analyze():
    try:
        files = {'file': open(IMAGE_PATH, 'rb')}
        response = requests.post(f"{BASE_URL}/analyze", files=files)
        if response.status_code == 200:
            print("\n✅ Analyze endpoint Check Passed!")
            print("Response Data:", response.json())
        else:
            print(f"\n❌ Analyze endpoint Check Failed: {response.status_code}")
            print(response.text)
    except FileNotFoundError:
        print(f"\n❌ Image file '{IMAGE_PATH}' not found.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    print(f"Testing API at {BASE_URL}...\n")
    test_root()
    test_analyze()
