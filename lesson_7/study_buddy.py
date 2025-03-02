import os
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


st.set_page_config(page_title="Study Buddy", layout="wide")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face API token is missing! Set it in GitHub Codespaces Secrets.")

# Set Hugging Face cache directory
cache_dir = "/tmp/huggingface"
os.environ["HF_HOME"] = cache_dir

# Load model
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=cache_dir, 
    torch_dtype="auto", 
    trust_remote_code=True
)
model.eval()


tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

st.title("Personalized Study Buddy")

# Padding token issue fix
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
        outputs = model.generate(**inputs, max_length=600, pad_token_id=tokenizer.pad_token_id)

    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Display answer
    st.success(f"Answer about {selected_topic}:")
    st.write(response)
