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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import getpass
import openai # For GPT-4o Zero-shot

TARGET_COLUMN = "cEXT" 
TEXT_COLUMN = "STATUS"
DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv"

def load_and_prepare_data(file_path, text_col, target_col):
    """Loads data from a CSV and prepares it for all experiments."""
    print("Loading and preparing data...")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    df = df.dropna(subset=[text_col, target_col])
    df['label'] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
    df_processed = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42, stratify=df_processed['label'])
    
    # For Hugging Face models
    train_val_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    train_val_split = train_val_dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': test_dataset
    })
    
    print("Data preparation complete.")
    return dataset_dict, train_df, test_df
# Load data once for all experiments
main_dataset_dict, train_df, test_df = load_and_prepare_data(DATA_FILE, TEXT_COLUMN, TARGET_COLUMN)
print("\nDataset structure for Transformer models:")
print(main_dataset_dict)


# ## Experiment 3: Zero-Shot Classification with GPT-4o
# This approach leverages a powerful, general-purpose model's existing knowledge. We simply ask it to classify the text based on a carefully crafted prompt, without any training data.
# --- GPT-4o Zero-Shot Setup ---
try:
    openai.api_key = getpass.getpass("Enter your OpenAI API key: ")
except Exception as e:
    print("Could not set OpenAI API key.", e)
    
def classify_with_gpt4o(text, trait):
    """Classifies a single text using GPT-4o with a zero-shot prompt."""
    prompt = f"""
    You are a psychology expert. Read the following text and determine if it indicates the author has the personality trait of '{trait}'.
    The trait '{trait}' is defined as being outgoing, talkative, and energetic.
    Respond with only the word 'yes' or 'no'.

    Text: "{text}"
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        return 1 if 'yes' in answer else 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1 # Return -1 for errors

# --- Evaluation ---
print("\nRunning GPT-4o zero-shot evaluation...")
# Note: This will make one API call per item in the test set. This can be slow and costly.
# We will run on a small sample.
sample_test_df = test_df.sample(n=10, random_state=42)
predictions_gpt = [classify_with_gpt4o(text, TARGET_COLUMN) for text in sample_test_df['text']]
true_labels_gpt = sample_test_df['label'].tolist()

# Filter out errors
valid_preds = [p for p, t in zip(predictions_gpt, true_labels_gpt) if p != -1]
valid_labels = [t for p, t in zip(predictions_gpt, true_labels_gpt) if p != -1]

if valid_labels:
    accuracy_gpt = accuracy_score(valid_labels, valid_preds)
    print(f"GPT-4o Zero-Shot Accuracy on 10 samples: {accuracy_gpt:.4f}")
else:
    print("Could not get any valid predictions from GPT-4o.")


# model_roberta_full.save_pretrained(OUTPUT_DIR_ROBERTA)
# tokenizer_roberta.save_pretrained(OUTPUT_DIR_ROBERTA)
# print(f"Model adapter and tokenizer saved to {OUTPUT_DIR_ROBERTA}")