import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType

## ---------------------------------------------------
## --- Configuration ---
## ---------------------------------------------------
ALL_TARGET_COLUMNS = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
TEXT_COLUMN = "STATUS"
DATA_FILE = "/users/PGS0218/julina/projects/LoRA_persona/data/mypersonality.csv"
MODEL_CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
BASE_OUTPUT_DIR = "/users/PGS0218/julina/projects/LoRA_persona/mypersonality/ckpt_llama_mlp_lora_final" 
my_token = "" # Your Hugging Face token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"Using Model: {MODEL_CHECKPOINT}")


## ---------------------------------------------------
## --- Custom Model Definition (Inheritance Method) ---
## ---------------------------------------------------
class LlamaForSequenceClassificationWithMLP(LlamaPreTrainedModel):
    """
    A custom Llama model for sequence classification that inherits from LlamaPreTrainedModel
    for full compatibility with the Hugging Face ecosystem. This is the most robust way
    to add a custom head.
    """
    def __init__(self, config, mlp_hidden_size=512, dropout_rate=0.2):
        super().__init__(config)
        # Store the number of labels, which is part of the config object
        self.num_labels = config.num_labels
        
        # The main Llama model body
        self.model = LlamaModel(config)
        
        # Our custom 2-layer MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, self.num_labels)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs, # Accept and forward any other arguments the Trainer might pass
    ):
        # Pass inputs through the base Llama model
        # The `output_hidden_states` argument is removed from here.
        # It will be passed automatically by the Trainer via **kwargs
        # when gradient_checkpointing is enabled.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Perform masked mean pooling to get a single sentence embedding
        hidden_states = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Pass the pooled sentence embedding to our classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


## ---------------------------------------------------
## --- Data Loading Function (no changes needed) ---
## ---------------------------------------------------
def load_and_prepare_data(file_path, text_col, target_col):
    """Loads data from a CSV and prepares it for all experiments."""
    print(f"\n--- Loading and preparing data for target: {target_col} ---")
    df = pd.read_csv(file_path, encoding='Windows-1252')
    df = df.dropna(subset=[text_col, target_col])
    
    if df[target_col].nunique() < 2:
        print(f"Warning: Not enough class diversity for {target_col}. Skipping this trait.")
        return None
        
    df['label'] = df[target_col].apply(lambda x: 1 if str(x).lower() == 'y' else 0)
    df_processed = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    
    train_df, _ = train_test_split(df_processed, test_size=0.1, random_state=42, stratify=df_processed['label'])
    
    train_val_dataset = Dataset.from_pandas(train_df)
    
    # Create a small validation set
    train_val_split = train_val_dataset.train_test_split(test_size=0.1, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test']
    })
    
    print("Data preparation complete.")
    return dataset_dict

## ---------------------------------------------------
## --- Tokenizer (loaded once) ---
## ---------------------------------------------------
print("Loading tokenizer...")
tokenizer_llama = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, token=my_token)
if tokenizer_llama.pad_token is None:
    tokenizer_llama.pad_token = tokenizer_llama.eos_token

## ---------------------------------------------------
## --- Main Execution Loop for All Traits ---
## ---------------------------------------------------
for target_trait in ALL_TARGET_COLUMNS:
    print("\n" + "="*80)
    print(f" PROCESSING TRAIT: {target_trait} ".center(80, "="))
    print("="*80)

    output_dir_trait = os.path.join(BASE_OUTPUT_DIR, f"{target_trait}")
    main_dataset_dict = load_and_prepare_data(DATA_FILE, TEXT_COLUMN, target_trait)
    if main_dataset_dict is None:
        continue

    # Initialize the custom model using the robust inheritance method
    print("\nInitializing custom LlamaForSequenceClassificationWithMLP model...")
    model = LlamaForSequenceClassificationWithMLP.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=my_token,
    )
    model.config.pad_token_id = tokenizer_llama.pad_token_id

    # Apply LoRA to the new model
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4, #16
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] 
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer_llama(examples["text"], truncation=True, max_length=128) # Keep max_length small
    tokenized_dataset = main_dataset_dict.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print("Configuring Trainer...")
    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=output_dir_trait,
            num_train_epochs=3,
            per_device_train_batch_size=2, # Use a small batch size
            gradient_accumulation_steps=8, # Accumulate to get effective batch size of 16
            gradient_checkpointing=True,   # Essential for saving memory
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            bf16=torch.cuda.is_available() # Use bfloat16 for performance
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))},
    )

    print(f"\nStarting fine-tuning for {target_trait} with robust inherited model...")
    trainer.train()

    print(f"Saving final model adapter and tokenizer to {output_dir_trait}")
    peft_model.save_pretrained(output_dir_trait)
    tokenizer_llama.save_pretrained(output_dir_trait)

print("\n" + "="*80)
print(" SCRIPT FINISHED: ALL TRAITS PROCESSED ".center(80, "="))
print("="*80)
