import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class PolicyDocumentLoader:
    """Handles loading of policy documents from various file formats"""

    # def __init__(self, policy_dir: str = "data/policies/raw"):
    def __init__(self, policy_dir: str = ""):
        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self._create_sample_policy_if_needed()

#     def _create_sample_policy_if_needed(self):
#         """Create sample policy file if directory is empty"""
#         sample_file = self.policy_dir / "policy.txt"
        
#         # Only create if file doesn't exist or is empty
#         if not sample_file.exists() or sample_file.stat().st_size == 0:
#             sample_content = """Travel Insurance Policy

# Coverage Details:
# - Medical expenses up to $100,000
# - Trip cancellation coverage
# - Lost baggage protection
# - 24/7 emergency assistance

# Exclusions:
# - Pre-existing conditions
# - Extreme sports
# - War zones"""
            
#             try:
#                 with open(sample_file, 'w', encoding='utf-8') as f:
#                     f.write(sample_content)
#                 print(f"Created sample policy at {sample_file}")
#             except Exception as e:
#                 print(f"Failed to create sample policy: {str(e)}")

    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document with validation"""
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Empty file: {file_path}")
            
        # Load based on extension
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
                
            return loader.load()
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
            raise

# Test the loader
try:
    loader = PolicyDocumentLoader()
    docs = loader.load_single_document("policy.txt")
    print(f"Successfully loaded {len(docs)} documents:")
    for doc in docs:
        print(f"- {doc.page_content[:100]}...")
        
except Exception as e:
    print(f"Failed to load documents: {str(e)}")