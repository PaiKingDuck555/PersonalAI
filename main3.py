from transformers import pipeline

# Load the fine-tuned model
model_path = "./dialogpt-finetuned-final"
generator = pipeline("text-generation", model=model_path, tokenizer=model_path)

# Generate a message
prompt = "I love you man!"
generated = generator(prompt, max_length=50, num_return_sequences=1)

print("Generated Message:", generated[0]['generated_text']) 
