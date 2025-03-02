import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

st.set_page_config(page_title="Study Buddy", layout="wide")
st.title("Personalized Study Buddy")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Fix padding token issue
tokenizer.pad_token = tokenizer.eos_token  

# Let user choose a study topic
topics = ["Mathematics", "Physics", "Biology", "Artificial Intelligence", "History"]
selected_topic = st.selectbox("Choose a Study Topic:", topics)

# Input field for the user
user_input = st.text_input(f"Ask a question about {selected_topic}:")

if user_input:
    # Create a context-aware prompt
    prompt = f"You are an expert in {selected_topic}. Answer the following question in a clear and concise manner:\n\nQuestion: {user_input}\n\nAnswer:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate response
    with st.spinner("Thinking... "):
        outputs = model.generate(**inputs, max_length=500, pad_token_id=tokenizer.pad_token_id)

    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Display answer
    st.success(f"Answer about {selected_topic}:")
    st.write(response)
