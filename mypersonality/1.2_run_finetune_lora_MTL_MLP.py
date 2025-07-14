import os, gc, sys
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup, 
)
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_pos_weight, create_unique_dir
from utils.trainer import Trainer


class ModelConfig:
    # LoRA Config
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    LORA_LR = 2e-4 

    # MLP Config
    HIDDEN_DIM = 512
    DROP_OUT = 0.3
    POOLING_METHOD = "mean"
    
    # Training Config
    EPOCHS = 8
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 8 # Effective batch size = 32
    LR = 2e-5 # Lower LR for the new MLP head
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 0 # NEW: Number of warmup steps for the scheduler
    EARLY_STOPPING_PATIENCE = 2
    TOKEN_MAX_LEN = 128
    OUTPUT_DIR = create_unique_dir('LLAMA_LoRA_MTL_MLP_v2')


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
        emb_dim = self.llama_peft.config.hidden_size

        # MODIFIED: Simplified MLP Head for stability and to prevent overfitting
        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(emb_dim, num_labels)
        )
        self.pooling_method = pooling_method
        print(f"This is a simplified MLP with {self.pooling_method} pooling method.")

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

        logits = self.mlp(pooled_embedding)
        return logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using config with differential LRs: LoRA={ModelConfig.LORA_LR}, MLP={ModelConfig.LR}")
    print(f'Output folder: {ModelConfig.OUTPUT_DIR}')

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df, val_df, test1_df, test2_df = load_and_prepare_all_data(0.1)
    
    # Preprocessing remains the same...
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
    test1_loader = DataLoader(dataset['test1'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)
    test2_loader = DataLoader(dataset['test2'], batch_size=ModelConfig.BATCH_SIZE, collate_fn=data_collator)

    model = LlamaMLPModel(
        model_checkpoint=Config.MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        hidden_dim=ModelConfig.HIDDEN_DIM,
        drop_out=ModelConfig.DROP_OUT,
        pooling_method=ModelConfig.POOLING_METHOD
    ).to(device)

    # NEW: Correctly set up differential learning rates
    optimizer_params = [
        {
            "params": [p for n, p in model.llama_peft.named_parameters() if "lora_" in n],
            "lr": ModelConfig.LORA_LR,  # Lower learning rate for LoRA params
            "weight_decay": 0.0  # Often LoRA params don't use weight decay
        },
        {
            "params": list(model.mlp.parameters()),
            # "params": [p for n,p in model.named_parameters() if 'mlp' in n],
            "lr": ModelConfig.LR,  # Higher learning rate for MLP
            "weight_decay": ModelConfig.WEIGHT_DECAY
        },
    ]
    optimizer = AdamW(optimizer_params)
    # optimizer = AdamW(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.WEIGHT_DECAY)

    # NEW: Set up the learning rate scheduler
    num_training_steps = ModelConfig.EPOCHS * (len(train_loader) // ModelConfig.GRAD_ACCUM_STEPS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=ModelConfig.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    pos_weight_tensor = compute_pos_weight(train_df, device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    best_val_f1 = 0
    patience_counter = 0
    print("\n--- Starting End-to-End Training ---")
    for epoch in range(ModelConfig.EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = criterion(logits, labels)
            
            # Normalize loss for accumulation
            loss = loss / ModelConfig.GRAD_ACCUM_STEPS
            total_train_loss += loss.item()
            loss.backward()

            # Perform optimizer step after accumulating gradients
            if (step + 1) % ModelConfig.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        
        val_res = Trainer.val_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{ModelConfig.EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_res['loss']:.4f}, Val F1: {val_res['f1_macro']:.4f}, Val Acc: {val_res['accuracy']:.4f}")

        if val_res['f1_macro'] > best_val_f1:
            best_val_f1 = val_res['f1_macro']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ModelConfig.OUTPUT_DIR, 'best_e2e_model.pt'))
            print(f"New best model saved with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= ModelConfig.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # optimal_thresholds = Trainer.find_optimal_thresholds(model, val_loader, device, Config.ALL_TARGET_COLUMNS)
    # print("Optimal thresholds per trait:", optimal_thresholds)

    print("\n--- Evaluating best model on test sets ---")
    model.load_state_dict(torch.load(os.path.join(ModelConfig.OUTPUT_DIR, 'best_e2e_model.pt')))

    Trainer.evaluate(model, test1_loader, device, Config.ALL_TARGET_COLUMNS, 'mypersonality')
    # Trainer.evaluate(model, test1_loader, device, Config.ALL_TARGET_COLUMNS, 'mypersonality', optimal_thresholds)
    Trainer.evaluate(model, test2_loader, device, Config.ALL_TARGET_COLUMNS, 'essay')
    print("\n SCRIPT FINISHED ".center(80, "="))


if __name__ == "__main__":
    main()
    # del model, model_lora, trainer
    gc.collect()
    torch.cuda.empty_cache()
