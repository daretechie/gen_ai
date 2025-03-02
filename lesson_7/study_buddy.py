import os
# import sys
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the system path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from util.model_loader import load_model
from dotenv import load_dotenv


# Set up the page
st.set_page_config(page_title="Study Buddy", layout="wide")
st.title("Personalized Study Buddy")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# model, tokenizer = load_model(hf_token=HF_TOKEN)

# Let user choose a study topic
topics = ["Mathematics", "Physics", "Biology", "Artificial Intelligence", "History"]
selected_topic = st.selectbox("Choose a Study Topic:", topics)

# Input field for the user
user_input = st.text_input(f"Ask a question about {selected_topic}:")

if user_input:
    # Create a context-aware prompt
    prompt = f"You are an expert in {selected_topic}. Answer the following question in a clear and concise manner:\n\nQuestion: {user_input}\n\nAnswer:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)

    # Generate response
    with st.spinner("Thinking... "):
        outputs = model.generate(**inputs, max_length=500, pad_token_id=tokenizer.pad_token_id)

    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Display answer
    st.success(f"Answer about {selected_topic}:")
    st.write(response)
