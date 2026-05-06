from artifacts import file_paths as fp
import pickle
from pathlib import Path

def posts():        
    with open(Path(fp.posts_path).expanduser(), 'rb') as f:
        data = pickle.load(f)
    return data

def rlm_rag_prompt():
    with open(Path(fp.prompts_path).expanduser(), 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("POSTS:")
    print(posts())
    
    print("\n")
    
    print("PROMPTS:")
    print(rlm_rag_prompt())