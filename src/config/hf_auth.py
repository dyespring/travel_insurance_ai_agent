from huggingface_hub import login

def authenticate_huggingface():
    """Authenticate with Hugging Face Hub"""
    try:
        login(token="token")  # Replace with your actual token
        print("Hugging Face authentication successful")
    except Exception as e:
        print(f"Hugging Face authentication failed: {str(e)}")
        raise