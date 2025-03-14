import os
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio


# st.set_page_config(page_title="Study Buddy", layout="wide")

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.run(asyncio.sleep(0))  # Ensure there's an event loop


# # Load environment variables
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     st.error("Hugging Face API token is missing! Set it in GitHub Codespaces Secrets.")

# # Set Hugging Face cache directory
# cache_dir = "/tmp/huggingface"
# os.environ["HF_HOME"] = cache_dir

# # Load model
# model_name = "microsoft/phi-1.5"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     cache_dir=cache_dir, 
#     torch_dtype="auto", 
#     trust_remote_code=True
# )
# model.eval()


# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

# st.title("Personalized Study Buddy")

# # Padding token issue fix
# tokenizer.pad_token = tokenizer.eos_token  

# # Let user choose a study topic
# topics = ["Mathematics", "Physics", "Biology", "Artificial Intelligence", "History"]
# selected_topic = st.selectbox("Choose a Study Topic:", topics)

# # Input field for the user
# user_input = st.text_input(f"Ask a question about {selected_topic}:")

# if user_input:
#     # Create a context-aware prompt
#     prompt = f"You are an expert in {selected_topic}. Answer the following question in a clear and concise manner:\n\nQuestion: {user_input}\n\nAnswer:"

#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

#     # Generate response
#     with st.spinner("Thinking... "):
#         # outputs = model.generate(**inputs, max_length=600, pad_token_id=tokenizer.pad_token_id)
#         outputs = model.generate(**inputs, max_length=300, pad_token_id=tokenizer.pad_token_id, temperature=0.7, top_p=0.9)

#     # Decode response
#     response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
#     # Display answer
#     st.success(f"Answer about {selected_topic}:")
#     st.write(response)


# ########################################################


# Set cache directory
cache_dir = "/tmp/huggingface"
os.environ["HF_HOME"] = cache_dir

# Load model and tokenizer
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype="auto", trust_remote_code=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Python Study Buddy 🤖")
st.write("Ask your Python programming questions, and I'll provide detailed responses!")

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Study Buddy:** {bot_msg}")

# User input
question = st.text_area("Enter your Python question:", key="input_area")

if st.button("Send"):
    if question.strip():
        # Add user question to chat history
        st.session_state.chat_history.append((question, ""))

        # Construct chat prompt
        chat_prompt = "You are an expert on the Python language.\n\n"
        for user_msg, bot_msg in st.session_state.chat_history:
            chat_prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        chat_prompt += f"User: {question}\nAssistant:"

        # Tokenize input
        inputs = tokenizer(chat_prompt, return_tensors="pt")

        # Generate response with fixed settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=512, 
                temperature=0.7, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True  # Enable sampling to avoid warnings
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's reply
        assistant_reply = response.split("Assistant:")[-1].strip()

        # Update chat history
        st.session_state.chat_history[-1] = (question, assistant_reply)

        # Streamlit will auto-refresh, so no need for experimental_rerun()
    else:
        st.warning("Please enter a question before clicking 'Send'.")
