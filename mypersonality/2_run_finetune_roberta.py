import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch

## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
# MODIFICATION: Define all target traits to loop through
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
TEXT_COLUMN = "STATUS"
DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv"
MODEL_CHECKPOINT_ROBERTA = "bert-base-uncased"
print(MODEL_CHECKPOINT_ROBERTA)

# MODIFICATION: Define a base directory for all RoBERTa outputs
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_roberta_full/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## ---------------------------------------------------
## --- Data Loading Function ---
## ---------------------------------------------------
def load_and_prepare_data(file_path, text_col, target_col):
    """Loads data from a CSV and prepares it for a specific target trait."""
    print(f"\n--- Loading and preparing data for target: {target_col} ---")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    df = df.dropna(subset=[text_col, target_col])
    
    # Check for sufficient class diversity before proceeding
    if df[target_col].nunique() < 2:
        print(f"Warning: Not enough class diversity for {target_col}. Skipping this trait.")
        return None
        
    df['label'] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
    df_processed = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    train_df, _ = train_test_split(df_processed, test_size=0.2, random_state=42, stratify=df_processed['label'])
    
    train_val_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    # The full test set isn't used in this script, but the split logic remains
    test_dataset = Dataset.from_pandas(df_processed, preserve_index=False)
    
    train_val_split = train_val_dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': test_dataset # Note: This 'test' set is the full processed df
    })
    
    print("Data preparation complete.")
    return dataset_dict

## ---------------------------------------------------
## --- Tokenizer (loaded once) ---
## ---------------------------------------------------
print("Loading RoBERTa tokenizer...")
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_ROBERTA)

## ---------------------------------------------------
## --- Main Execution Loop for All Traits ---
## ---------------------------------------------------
for target_trait in ALL_TARGET_COLUMNS:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    # 1. Set up dynamic output directory for the current trait
    output_dir_trait = os.path.join(BASE_OUTPUT_DIR, f"{target_trait}")
    print(f"Output directory for this run: {output_dir_trait}")

    # 2. Load and prepare data for the current trait
    main_dataset_dict = load_and_prepare_data(DATA_FILE, TEXT_COLUMN, target_trait)
    if main_dataset_dict is None:
        continue # Skip to the next trait if data is insufficient

    # 3. Re-initialize the RoBERTa model from the base checkpoint for each trait
    # This is critical to ensure each fine-tuning run is independent.
    print(f"\nInitializing {MODEL_CHECKPOINT_ROBERTA} model...")
    model_roberta_full = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_ROBERTA, num_labels=2).to(device)
    print(f"{MODEL_CHECKPOINT_ROBERTA} has {model_roberta_full.num_parameters():,} total parameters to be fine-tuned.")

    # 4. Tokenize the specific dataset for the current trait
    print("\nTokenizing dataset...")
    def tokenize_function_roberta(examples):
        return tokenizer_roberta(examples["text"], truncation=True, max_length=512)
    tokenized_dataset_roberta = main_dataset_dict.map(tokenize_function_roberta, batched=True).remove_columns(["text"])

    # 5. Set up the Trainer for the current trait
    print("Configuring Trainer...")
    trainer_roberta = Trainer(
        model=model_roberta_full,
        args=TrainingArguments(
            output_dir=output_dir_trait, # Use the dynamic output directory
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5, # A slightly lower learning rate is often better for full fine-tuning
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(), # Use mixed precision for speed on GPU
        ),
        train_dataset=tokenized_dataset_roberta["train"],
        eval_dataset=tokenized_dataset_roberta["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_roberta),
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))},
    )
    
    # 6. Train and Save the model for the current trait
    print(f"\nStarting RoBERTa-Large full fine-tuning for {target_trait}...")
    trainer_roberta.train() # Uncomment to run training
    print(f"RoBERTa full fine-tuning would be run here for {target_trait}.")

    print(f"Saving model and tokenizer to {output_dir_trait}")
    model_roberta_full.save_pretrained(output_dir_trait)
    tokenizer_roberta.save_pretrained(output_dir_trait)

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)