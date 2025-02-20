import requests
import json

def test_prediction():
    url = "http://localhost:8000/predict"
    data = {
        "texts": ["example text to classify", "another example"]
    }
    
    try:
        print("Sending request with texts:", json.dumps(data["texts"], indent=2))
        print("\nMaking API call...")
        
        response = requests.post(url, json=data)
        
        print("\nStatus Code:", response.status_code)
        print("\nFull Response:")
        print(json.dumps(response.json(), indent=2))
        
        # Print individual predictions
        print("\nPredictions breakdown:")
        for i, (text, pred) in enumerate(zip(data["texts"], response.json()["predictions"])):
            print(f"\nText {i+1}: '{text}'")
            print(f"Class 0 probability: {pred['class_0_prob']:.4f}")
            print(f"Class 1 probability: {pred['class_1_prob']:.4f}")
            print(f"Predicted class: {pred['predicted_class']}")
            
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_prediction() 