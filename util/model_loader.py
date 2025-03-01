import os
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv


def load_model(model_dir="./util/microsoft_phi2", hf_token=None):
    # Check if the model directory exists and contains the required files
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]  # Add other necessary files if needed
    if not os.path.exists(model_dir) or not all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
        print("Downloading and saving the model ...")
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print("Model saved in local directory!")
    else:
        print("Loading model from local directory...")
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print("Model loaded from local directory!")

    # Authenticate with Hugging Face if a token is provided
    if hf_token:
        login(hf_token)
        print("Hugging Face authentication successful!")

    return model, tokenizer

# def load_model(model_dir="./util/microsoft_phi_2", hf_token=None):
#     if not os.path.exists(model_dir):
#         print("Downloading and saving the model ...")
#         model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
#         tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

#         # Save model and tokenizer
#         model.save_pretrained(model_dir)
#         tokenizer.save_pretrained(model_dir)
#         print("Model saved in local directory!")
#     else:
#         print("Loading model from local directory...")
#         model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", trust_remote_code=True)
#         tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#         print("Model loaded from local directory!")

#     # Authenticate with Hugging Face if a token is provided
#     if hf_token:
#         login(hf_token)
#         print(" Hugging Face authentication successful!")

#     return model, tokenizer

# Load the model

# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")  
# model, tokenizer = load_model(hf_token=HF_TOKEN)

