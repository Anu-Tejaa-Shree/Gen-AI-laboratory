from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("Loading GPT-2 model...")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# Input prompt
prompt = input("Enter a prompt: ")

# Convert prompt into tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Convert tokens back into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n========== GENERATED TEXT ==========\n")
print(generated_text)