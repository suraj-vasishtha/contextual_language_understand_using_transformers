import streamlit as st
from transformers import pipeline

# Initialize the classifier
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# Create the web interface
st.title("Simple Text Classifier")

# Text input
text = st.text_area("Enter text to classify:", "I love this!")

if st.button("Classify"):
    # Get prediction
    result = classifier(text)[0]
    
    # Show results
    st.write("### Result:")
    st.write(f"Label: {result['label']}")
    st.write(f"Confidence: {result['score']:.2%}") 