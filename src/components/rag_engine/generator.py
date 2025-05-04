from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

class ResponseGenerator:
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        """
        Initialize the response generator model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def generate(self, prompt: str, max_length: int = 200, **kwargs) -> str:
        """
        Generate a response based on the given prompt.
        
        Args:
            prompt: Input prompt for generation
            max_length: Maximum length of generated response
            kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            **kwargs
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()