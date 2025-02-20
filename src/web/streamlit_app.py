import streamlit as st
import requests
import json
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def check_api_status() -> bool:
    """Check if the API server is running"""
    ports = [8000, 8001, 8002, 8003, 8004]
    for port in ports:
        try:
            url = f"http://localhost:{port}/health"
            response = requests.get(url)
            if response.status_code == 200:
                st.session_state['api_port'] = port
                return True
        except requests.exceptions.ConnectionError:
            continue
    return False

def make_prediction(text: str) -> dict:
    """Make a prediction using the API"""
    if 'api_port' not in st.session_state:
        if not check_api_status():
            st.error("API server is not running. Please start the server first.")
            st.info("Run 'python -m src.deploy' in a separate terminal.")
            return None
    
    try:
        url = f"http://localhost:{st.session_state['api_port']}/predict"
        data = {"texts": [text]}
        
        response = requests.post(url, json=data)
        return response.json()["predictions"][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Make sure the API server is running and try again.")
        return None

def main():
    st.title("Text Classification Interface")
    st.write("Enter text to classify it using our transformer model.")
    
    # Check API status
    if not check_api_status():
        st.error("⚠️ API server is not running!")
        st.info("Please follow these steps:")
        st.code("1. Open a new terminal\n2. Run: python -m src.deploy")
        st.warning("Keep the API server terminal open while using this interface.")
        return
    
    # Show API status
    st.success("✅ API server is connected!")
    
    # Text input
    text_input = st.text_area("Enter text:", height=100)
    
    # Prediction button
    if st.button("Classify Text"):
        if text_input:
            with st.spinner("Making prediction..."):
                prediction = make_prediction(text_input)
                
                if prediction:
                    # Create columns for displaying results
                    col1, col2 = st.columns(2)
                    
                    # Display prediction
                    with col1:
                        st.subheader("Prediction")
                        predicted_class = prediction["predicted_class"]
                        st.write(f"Class {predicted_class}")
                        
                        # Add confidence bar
                        confidence = max(prediction["class_0_prob"], 
                                      prediction["class_1_prob"])
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence:.2%}")
                    
                    # Display probabilities
                    with col2:
                        st.subheader("Class Probabilities")
                        st.write("Class 0:", f"{prediction['class_0_prob']:.2%}")
                        st.write("Class 1:", f"{prediction['class_1_prob']:.2%}")
                        
                    # Add explanation
                    st.subheader("Explanation")
                    st.write("""
                    - Class 0: First category
                    - Class 1: Second category
                    
                    The confidence score indicates how sure the model is about its prediction.
                    """)
        else:
            st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main() 