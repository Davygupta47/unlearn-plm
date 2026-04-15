import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen1.5-0.5B"
save_directory = "models/Qwen1.5-0.5B"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Created directory: {save_directory}")

print(f"Downloading and loading {model_name}...")

try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Saving to {save_directory}...")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    print("Success! Model and Tokenizer are now stored locally.")

except Exception as e:
    print(f"An error occurred: {e}")