import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel, # Using the base model without a specific head
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn

## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
TEXT_COLUMN = "STATUS"
DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv"
MODEL_CHECKPOINT_BERT = "bert-base-uncased"
print(f"Using base model: {MODEL_CHECKPOINT_BERT}")

# MODIFICATION: Updated base directory for the new MLP model architecture
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_bert_mlp/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## ---------------------------------------------------
## --- Custom Model Definition: BERT + 2-Layer MLP ---
## ---------------------------------------------------
class BertMLPClassifier(nn.Module):
    """
    A custom model that places a 2-layer MLP classifier on top of a pretrained BERT model.
    It uses the [CLS] token's pooled output for classification.
    The entire model (BERT base + new MLP head) is fine-tuned.
    """
    def __init__(self, base_model, num_labels, mlp_hidden_dim=512, dropout_prob=0.2):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = base_model
        self.config = base_model.config

        # Define the 2-Layer MLP head using nn.Sequential
        self.mlp_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_dim, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 1. Get outputs from the base BERT model
        transformer_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Use the 'pooler_output', which is the [CLS] token's hidden state
        # passed through a Linear layer and Tanh activation. It's designed for
        # sequence-level classification.
        pooled_output = transformer_outputs.pooler_output

        # 3. Pass the pooled output through the MLP head to get logits
        logits = self.mlp_head(pooled_output)

        # 4. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def save_pretrained(self, save_directory):
        """Saves the fine-tuned base model and the custom MLP head."""
        print(f"Saving model components to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        # Save the fine-tuned base model
        self.base_model.save_pretrained(save_directory)
        # Save the custom MLP head's state dict
        torch.save(self.mlp_head.state_dict(), os.path.join(save_directory, 'custom_mlp_head.pth'))


## ---------------------------------------------------
## --- Data Loading Function (no changes) ---
## ---------------------------------------------------
def load_and_prepare_data(file_path, text_col, target_col):
    """Loads data from a CSV and prepares it for a specific target trait."""
    print(f"\n--- Loading and preparing data for target: {target_col} ---")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    df = df.dropna(subset=[text_col, target_col])
    
    if df[target_col].nunique() < 2:
        print(f"Warning: Not enough class diversity for {target_col}. Skipping this trait.")
        return None
        
    df['label'] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
    df_processed = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    train_df, _ = train_test_split(df_processed, test_size=0.1, random_state=42, stratify=df_processed['label'])
    
    train_val_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(df_processed, preserve_index=False)
    
    train_val_split = train_val_dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': test_dataset
    })
    
    print("Data preparation complete.")
    return dataset_dict

## ---------------------------------------------------
## --- Tokenizer (loaded once) ---
## ---------------------------------------------------
print(f"Loading {MODEL_CHECKPOINT_BERT} tokenizer...")
tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_BERT)

## ---------------------------------------------------
## --- Main Execution Loop for All Traits ---
## ---------------------------------------------------
for target_trait in ALL_TARGET_COLUMNS:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    output_dir_trait = os.path.join(BASE_OUTPUT_DIR, f"{target_trait}")
    print(f"Output directory for this run: {output_dir_trait}")

    main_dataset_dict = load_and_prepare_data(DATA_FILE, TEXT_COLUMN, target_trait)
    if main_dataset_dict is None:
        continue

    # MODIFICATION: Initialize base BERT, then wrap with custom MLP head
    print(f"\nInitializing base {MODEL_CHECKPOINT_BERT} model...")
    base_model = AutoModel.from_pretrained(MODEL_CHECKPOINT_BERT)
    
    print("Initializing custom BERT-MLP classifier...")
    model_mlp = BertMLPClassifier(base_model=base_model, num_labels=2).to(device)
    
    total_params = sum(p.numel() for p in model_mlp.parameters() if p.requires_grad)
    print(f"BERT+MLP model has {total_params:,} total parameters to be fine-tuned.")

    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer_bert(examples["text"], truncation=True, max_length=256) # Using a reduced max_length to save memory
    tokenized_dataset = main_dataset_dict.map(tokenize_function, batched=True).remove_columns(["text"])

    print("Configuring Trainer...")
    trainer = Trainer(
        model=model_mlp, # MODIFICATION: Use the custom MLP model
        args=TrainingArguments(
            output_dir=output_dir_trait,
            num_train_epochs=3,
            per_device_train_batch_size=4,      # Kept small to manage memory
            gradient_accumulation_steps=4,      # Simulates a batch size of 16
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_bert),
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))},
    )
    
    # Clear cache before starting training
    torch.cuda.empty_cache()
    print(f"\nStarting BERT+MLP full fine-tuning for {target_trait}...")
    trainer.train()

    print(f"Saving final model and tokenizer to {output_dir_trait}")
    model_mlp.save_pretrained(output_dir_trait)
    tokenizer_bert.save_pretrained(output_dir_trait)

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)