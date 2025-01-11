from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import Dataset
import logging 
import gc
import os 
import torch


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(file_path, tokenizer):
    with open(file_path, "r", encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
   
    logger.info(f"Loaded {len(lines)} lines from dataset")
    raw_data = {"text": lines}
    dataset = Dataset.from_dict(raw_data)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Dataset size after tokenization: {len(tokenized_dataset)}")
    return tokenized_dataset

def main():
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "openai-community/gpt2",  # Using medium instead of large
            use_fast=True,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            trust_remote_code=True
        ).to(device)
        
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Load and preprocess dataset
    file_path = "messages.txt"
    try:
        dataset = preprocess_data(file_path, tokenizer)
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments with modified parameters
    training_args = TrainingArguments(
        output_dir="./dialogpt-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size
        gradient_accumulation_steps=4,   # Added gradient accumulation
        warmup_steps=100,               # Added warmup steps
        logging_dir="./logs",
        logging_steps=10,               # More frequent logging
        save_steps=100,                # More frequent saving
        save_total_limit=2,
        learning_rate=5e-5,            # Adjusted learning rate
        weight_decay=0.01,             # Added weight decay
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        report_to="none"               # Disable wandb reporting
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Start training with progress monitoring
    logger.info("Starting training...")
    try:
        torch.save(model.state_dict(), 'checkpoint.pth')
        model.load_state_dict(torch.load('checkpoint.pth'))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save the model
        model.save_pretrained("./dialogpt-finetuned-final")
        tokenizer.save_pretrained("./dialogpt-finetuned-final")
        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")

if __name__ == "__main__":
    main()