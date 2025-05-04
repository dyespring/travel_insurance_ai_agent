import os
from pathlib import Path
from embedder import DocumentEmbedder

def check_file_permissions(filepath):
    if not os.access(filepath, os.R_OK):
        print(f"Permission error: Cannot read {filepath}")
        print("Try running as administrator or checking file permissions")
        return False
    return True

def main():
    embedder = DocumentEmbedder()
    policy_dir = Path("data") / "policies" / "raw"
    
    # Create directory if needed
    policy_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample policy file if none exists
    sample_file = policy_dir / "policy.txt"
    if not sample_file.exists():
        try:
            with open(sample_file, 'w') as f:
                f.write("Sample travel insurance policy covering medical expenses up to $100,000")
            print(f"Created sample policy file at {sample_file}")
        except PermissionError:
            print(f"Cannot create sample file at {sample_file} - check directory permissions")
            return
    
    # Verify permissions before processing
    if not check_file_permissions(sample_file):
        return
    
    try:
        chunks, embeddings = embedder.process_policy_files()
        print(f"\nSuccessfully processed {len(chunks)} chunks")
        print(f"Sample chunk: {chunks[0][:50]}...")
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Possible solutions:")
        print("1. Check file permissions in data/policies/raw/")
        print("2. Run IDE/text editor as administrator")
        print("3. Verify files exist with supported extensions (.pdf, .txt, .md)")

if __name__ == "__main__":
    main()