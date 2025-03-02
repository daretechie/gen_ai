import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix for tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Hugging Face cache directory
cache_dir = "/tmp/huggingface"
os.environ["HF_HOME"] = cache_dir

# Load model
model_name = "microsoft/phi-2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# Set up app title and description
st.title("Chat with Phi-1.5")
st.write("Hello! I'm your chatbot powered by Microsoft's Phi-1.5 model. Start chatting with me!")

# Initialize session state for chat history and conversation turn tracking
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Load model (cached after first run)
tokenizer, model = load_model()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("What would you like to discuss?")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate bot response
    try:
        # Tokenize the new user input
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        
        # Append the new user input to the chat history
        bot_input_ids = new_user_input_ids
        
        # If there is a chat history, append it to the input
        if st.session_state.chat_history_ids is not None:
            bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
        
        # Generate response with proper context
        with torch.no_grad():
            output_ids = model.generate(
                bot_input_ids,
                max_length=2000,  # Increased max length
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Avoid repetition
                attention_mask=torch.ones_like(bot_input_ids)  # Fix missing attention mask
            )
        
        # Save the generated response for next round
        st.session_state.chat_history_ids = output_ids
        
        # Decode response and remove input tokens
        bot_response = tokenizer.decode(output_ids[0][bot_input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Handle empty response
        if not bot_response.strip():
            bot_response = "I'm not sure how to respond to that. Could you try asking something else?"
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again with a different question."
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a reset button to clear chat history
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.chat_history_ids = None
    st.experimental_rerun()
