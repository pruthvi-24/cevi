import requests
import sys
import time

BASE_URL = "http://127.0.0.1:8001"
IMAGE_PATH = "biryani.jpg"

def test_analyze_randomness():
    print(f"Testing API at {BASE_URL} for execution randomness...\n")
    
    previous_dish = None
    different_count = 0
    attempts = 5

    for i in range(attempts):
        try:
            # We use the same image, but the mock should return different dishes now
            files = {'file': open(IMAGE_PATH, 'rb')}
            response = requests.post(f"{BASE_URL}/analyze", files=files)
            
            if response.status_code == 200:
                data = response.json()
                current_dish = data['data']['dish']
                print(f"Attempt {i+1}: Got dish '{current_dish}'")
                
                if previous_dish and current_dish != previous_dish:
                    different_count += 1
                previous_dish = current_dish
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            break
        
        # files.close() is handled by context manager if we used 'with', but here we just open fresh
    
    if different_count > 0:
        print("\n✅ Validated: API returns different/random dishes (simulation works).")
    else:
        print("\n⚠️ Warning: All responses were the same. Randomness might not be working or we got unlucky.")

if __name__ == "__main__":
    test_analyze_randomness()
