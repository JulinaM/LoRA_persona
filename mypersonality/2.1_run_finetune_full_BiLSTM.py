import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel, # MODIFICATION: Changed from AutoModelForSequenceClassification
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
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
MODEL_CHECKPOINT_BERT = "bert-base-uncased" # Using a more specific name for clarity
print(f"Using base model: {MODEL_CHECKPOINT_BERT}")

# MODIFICATION: Updated base directory for the new model architecture
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_bert_bilstm/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## ---------------------------------------------------
## --- Custom Model Definition: BERT + BiLSTM ---
## ---------------------------------------------------
class BertBiLSTMClassifier(nn.Module):
    """
    A custom model that places a BiLSTM classifier on top of a pretrained BERT model.
    The entire model (BERT base + new layers) is fine-tuned.
    """
    def __init__(self, base_model, num_labels, lstm_hidden_size=256, lstm_dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.base_model = base_model # The fully-trainable BERT model
        self.config = base_model.config # For compatibility with Trainer

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size, # For bert-base, this is 768
            hidden_size=lstm_hidden_size,
            num_layers=1,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        # Classifier Head
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels) # *2 for bidirectional

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 1. Get hidden states from the base BERT model
        transformer_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs.last_hidden_state

        # 2. Pass hidden states to BiLSTM
        lstm_output, (h_n, c_n) = self.lstm(hidden_states)

        # 3. Concatenate final forward and backward hidden states
        final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # 4. Pass through dropout and classifier
        pooled_output = self.dropout(final_hidden_state)
        logits = self.classifier(pooled_output)

        # 5. Calculate loss if labels are provided
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
        """Saves the fine-tuned base model and the custom BiLSTM head."""
        print(f"Saving model components to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        # Save the fine-tuned base model (it's a PreTrainedModel, so this works)
        self.base_model.save_pretrained(save_directory)
        # Save the custom classification head's state dict
        torch.save(self.lstm.state_dict(), os.path.join(save_directory, 'lstm.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, 'classifier.pth'))


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
    
    train_df, _ = train_test_split(df_processed, test_size=0.2, random_state=42, stratify=df_processed['label'])
    
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

    # MODIFICATION: Initialize base BERT, then wrap with custom BiLSTM head
    print(f"\nInitializing base {MODEL_CHECKPOINT_BERT} model...")
    base_model = AutoModel.from_pretrained(MODEL_CHECKPOINT_BERT)
    
    print("Initializing custom BERT-BiLSTM classifier...")
    model_bilstm = BertBiLSTMClassifier(base_model=base_model, num_labels=2).to(device)
    
    total_params = sum(p.numel() for p in model_bilstm.parameters() if p.requires_grad)
    print(f"BERT+BiLSTM model has {total_params:,} total parameters to be fine-tuned.")

    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer_bert(examples["text"], truncation=True, max_length=512)
    tokenized_dataset = main_dataset_dict.map(tokenize_function, batched=True).remove_columns(["text"])

    print("Configuring Trainer...")
    trainer = Trainer(
        model=model_bilstm, # MODIFICATION: Use the custom model
        args=TrainingArguments(
            output_dir=output_dir_trait,
            num_train_epochs=5,
            per_device_train_batch_size=16,
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
    
    print(f"\nStarting BERT+BiLSTM full fine-tuning for {target_trait}...")
    trainer.train()

    print(f"Saving final model and tokenizer to {output_dir_trait}")
    # MODIFICATION: The custom save_pretrained method handles all components
    model_bilstm.save_pretrained(output_dir_trait)
    tokenizer_bert.save_pretrained(output_dir_trait)

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)