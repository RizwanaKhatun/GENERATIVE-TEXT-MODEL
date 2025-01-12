from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate text based on a prompt
def generate_text(prompt, max_length=100):
    # Encode the prompt text
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text based on the prompt
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,  # Avoid repeating n-grams
        temperature=0.7,  # Control randomness
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling
        do_sample=True,  # Sampling vs. greedy decoding
        pad_token_id=tokenizer.eos_token_id  # Padding token
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The future of artificial intelligence is"
generated_paragraph = generate_text(prompt)
print(generated_paragraph)