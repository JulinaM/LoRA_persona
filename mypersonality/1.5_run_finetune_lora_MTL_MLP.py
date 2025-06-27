import os, gc, sys
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_mtl_metrics


class ModelConfig:
    # LoRA Config
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.1
    
    # MLP Config
    HIDDEN_DIM = 512
    DROP_OUT = 0.5
    POOLING_METHOD = "max"
    
    # Training Config
    EPOCHS = 5
    BATCH_SIZE = 4 # This is the per-device batch size
    GRAD_ACCUM_STEPS = 8 # Effective batch size = 32
    LR = 0.00001 # A conservative learning rate for end-to-end training
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 2
    TOKEN_MAX_LEN = 128 # Config.TOKEN_MAX_LEN 512
    BASE_OUTPUT_DIR = Config.BASE_OUTPUT_DIR +'/LLAMA_LoRA_MTL_MLP/'


class LlamaMLPModel(torch.nn.Module):
    def __init__(self, model_checkpoint, num_labels, hidden_dim, drop_out=0.5, pooling_method="mean"):
        super().__init__()
        self.llama_base = AutoModel.from_pretrained(model_checkpoint, token=Config.get_hf_token(), torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=ModelConfig.LORA_R,
            lora_alpha=ModelConfig.LORA_ALPHA,
            lora_dropout=ModelConfig.LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.llama_peft = get_peft_model(self.llama_base, lora_config)
        self.llama_peft.print_trainable_parameters()

        hidden_dim = self.llama_peft.config.hidden_size
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(hidden_dim//2, num_labels)
        )
        self.pooling_method = pooling_method
        print(f"This is two layer MLP with {self.pooling_method} pooling method.")

    def forward(self, input_ids, attention_mask):
        outputs = self.llama_peft(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if self.pooling_method == 'mean':
            last_hidden_state = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled_embedding = sum_embeddings / sum_mask
        
        elif self.pooling_method == 'max':
            last_hidden_state = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            masked_embeddings = last_hidden_state * mask + (1 - mask) * -1e9  # Set padding to -inf
            pooled_embedding = torch.max(masked_embeddings, dim=1).values  # [batch, hidden_dim]
        
        else:  
            pooled_embedding = outputs.last_hidden_state[:, -1, :].float()

        logits = self.mlp_head(pooled_embedding)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using config: {ModelConfig.LR}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df, val_df, test1_df, test2_df = load_and_prepare_all_data(0.1)
    print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, and test sets.")
    
    labels_df = train_df[Config.ALL_TARGET_COLUMNS]
    pos_counts = labels_df.sum()
    neg_counts = len(labels_df) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8) # Clamp to avoid division by zero if a class has 0 positive samples
    pos_weight_tensor = torch.tensor(pos_weight.values, dtype=torch.float).to(device)
    print("\nCalculated positive class weights for weighted loss:")
    print(pos_weight)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, max_length=ModelConfig.TOKEN_MAX_LEN)
        labels = np.array([examples[col] for col in Config.ALL_TARGET_COLUMNS])
        tokenized_inputs['labels'] = labels.T.astype(np.float32)
        return tokenized_inputs

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'validation': Dataset.from_pandas(val_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'test1': Dataset.from_pandas(test1_df).map(preprocess_function, batched=True, remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
        'test2': Dataset.from_pandas(test2_df).map(preprocess_function, batched=True,remove_columns=["text"] + Config.ALL_TARGET_COLUMNS),
    })

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(dataset["train"], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)

    model = LlamaMLPModel(
        model_checkpoint=Config.MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        hidden_dim=ModelConfig.HIDDEN_DIM,
        drop_out=ModelConfig.DROP_OUT,
        pooling_method=ModelConfig.POOLING_METHOD
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.WEIGHT_DECAY)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_f1 = 0
    patience_counter = 0
    print("\n--- Starting End-to-End Training ---")
    for epoch in range(ModelConfig.EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        all_val_preds, all_val_labels = [], []
        v_total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                v_loss = criterion(logits, labels)
                v_total_loss += v_loss.item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        val_preds = torch.cat(all_val_preds).numpy()
        val_labels = torch.cat(all_val_labels).numpy()
        val_f1 = f1_score(val_labels, val_preds, average='micro')
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{ModelConfig.EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {v_total_loss / len(val_loader):.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs(ModelConfig.BASE_OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ModelConfig.BASE_OUTPUT_DIR, 'best_e2e_model.pt'))
            print(f"New best model saved with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= ModelConfig.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final Evaluation
    print("\n--- Evaluating best model on test sets ---")
    model.load_state_dict(torch.load(os.path.join(ModelConfig.BASE_OUTPUT_DIR, 'best_e2e_model.pt')))
    model.eval()

    test_loaders = {
        "mypersonality": DataLoader(dataset['test1'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator),
        "Essay": DataLoader(dataset['test2'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)
    }

    with torch.no_grad():
        for split_name, loader in test_loaders.items():
            all_test_preds, all_test_labels = [], []
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_test_preds.append(preds.cpu())
                all_test_labels.append(labels.cpu())
            
            test_preds = torch.cat(all_test_preds).numpy()
            true_labels = torch.cat(all_test_labels).numpy()

            f1_micro = f1_score(true_labels, test_preds, average='micro')
            accuracy = accuracy_score(true_labels, test_preds)

            print("\n" + "-"*50)
            print(f"Final Test Set '{split_name}' Evaluation Results (Overall):")
            print(f"  Subset Accuracy (Exact Match): {accuracy:.4f}")
            print(f"  F1 Score (micro): {f1_micro:.4f}")
            
            print("\nPer-Trait Test Results:")
            for i, trait in enumerate(Config.ALL_TARGET_COLUMNS):
                trait_acc = accuracy_score(true_labels[:, i], test_preds[:, i])
                trait_f1 = f1_score(true_labels[:, i], test_preds[:, i], average='binary')
                
                print(f"  --- {trait} ---")
                print(f"    Accuracy: {trait_acc:.4f}")
                print(f"    F1 Score: {trait_f1:.4f}")
            print("-"*50)

    print("\n SCRIPT FINISHED ".center(80, "="))

if __name__ == "__main__":
    main()
