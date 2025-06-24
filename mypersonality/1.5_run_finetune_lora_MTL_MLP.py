import os, gc, json, sys
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0,'/users/PGS0218/julina/projects/LoRA_persona')
from utils.lora_utils import Config, load_and_prepare_all_data, compute_mtl_metrics, create_out_dir


class LoRA_Config:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )
    
    training_args = TrainingArguments(
        output_dir=Config.BASE_OUTPUT_DIR+'/llama_lora_mtl_mlp/',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_total_limit=1,
        group_by_length=True,
    )

class MLP_Config:
    INPUT_DIM = 4096  # Llama-3-8B hidden size
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(Config.ALL_TARGET_COLUMNS)
    EPOCHS = 20
    LR = 1e-4
    WEIGHT_DECAY = 2e-5
    EARLY_STOPPING_PATIENCE = 3
    BATCH_SIZE = 32


class MLPClassifier(torch.nn.Module):
    """A simple two-layer MLP for multi-label classification."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x # Return raw logits

def get_embeddings(model, dataset, tokenizer, device):
    print("Extracting embeddings...")
    model.eval()
    all_embeddings = []
    
    temp_trainer = Trainer(model=model, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    dataloader = temp_trainer.get_eval_dataloader(dataset)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model.base_model.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            mask = batch['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Model: {Config.MODEL_CHECKPOINT}")

    tokenizer_llama = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT, token=Config.get_hf_token())
    if tokenizer_llama.pad_token is None:
        tokenizer_llama.pad_token = tokenizer_llama.eos_token

    trainval_df, test1_df, test2_df = load_and_prepare_all_data(0.15)
    train_df, val_df = train_test_split(trainval_df, test_size=0.15, random_state=42)
    print(f"\nData split into {len(train_df)} train, {len(val_df)} validation, and test sets.")


    raw_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test1': Dataset.from_pandas(test1_df),
        'test2': Dataset.from_pandas(test2_df)
    })
        
    def tokenize_function(examples):
        tokenized_inputs = tokenizer_llama(examples['text'], truncation=True, max_length=128)
        labels = np.array([examples[col] for col in Config.ALL_TARGET_COLUMNS])
        tokenized_inputs['labels'] = labels.T.astype(np.float32).tolist()
        return tokenized_inputs
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset['train'].column_names)

    print("\n--- STAGE 1: Fine-tuning Llama-LoRA for Multi-Task Learning ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=len(Config.ALL_TARGET_COLUMNS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer_llama.pad_token_id
    model_lora = get_peft_model(model, LoRA_Config.lora_config)
    model_lora.print_trainable_parameters()

    trainer = Trainer(
        model=model_lora,
        args=LoRA_Config.training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_llama),
        compute_metrics=compute_mtl_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    print("\n--- STAGE 2: MLP Training and Evaluation ---")
    fine_tuned_model = trainer.model

    train_embeddings = get_embeddings(fine_tuned_model, tokenized_dataset['train'], tokenizer_llama, device)
    val_embeddings = get_embeddings(fine_tuned_model, tokenized_dataset['validation'], tokenizer_llama, device)
    test1_embeddings = get_embeddings(fine_tuned_model, tokenized_dataset['test1'], tokenizer_llama, device)
    test2_embeddings = get_embeddings(fine_tuned_model, tokenized_dataset['test2'], tokenizer_llama, device)
    
    # Get corresponding labels
    train_labels = torch.tensor(tokenized_dataset['train']['labels'])
    val_labels = torch.tensor(tokenized_dataset['validation']['labels'])
    test1_labels = torch.tensor(tokenized_dataset['test1']['labels'])
    test2_labels = torch.tensor(tokenized_dataset['test2']['labels'])

    # Create PyTorch DataLoaders
    train_data = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_data, batch_size=MLP_Config.BATCH_SIZE, shuffle=True)
    val_data = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_data, batch_size=MLP_Config.BATCH_SIZE)

    mlp_model = MLPClassifier(input_dim=MLP_Config.INPUT_DIM, hidden_dim=MLP_Config.HIDDEN_DIM, output_dim=MLP_Config.OUTPUT_DIM ).to(device)
    optimizer = AdamW(mlp_model.parameters(), lr=MLP_Config.LR, weight_decay=MLP_Config.WEIGHT_DECAY)
    criterion = BCEWithLogitsLoss()

    best_val_f1 = 0
    patience_counter = 0

    print("Training MLP classifier...")
    for epoch in range(MLP_Config.EPOCHS):
        mlp_model.train()
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        mlp_model.eval()
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = mlp_model(embeddings)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())
        
        val_preds = torch.cat(all_val_preds).numpy()
        val_labels = torch.cat(all_val_labels).numpy()
        val_f1 = f1_score(val_labels, val_preds, average='micro')

        if (epoch + 1) % 10 == 0:
            print(f'MLP Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(mlp_model.state_dict(), os.path.join(Config.BASE_OUTPUT_DIR, 'best_mlp_model.pt'))
        else:
            patience_counter += 1

        if patience_counter >= MLP_Config.EARLY_STOPPING_PATIENCE:
            print(f"MLP early stopping at epoch {epoch+1}")
            break

    # Final Evaluation
    print("\n--- Evaluating best MLP model on test sets ---")
    mlp_model.load_state_dict(torch.load(os.path.join(Config.BASE_OUTPUT_DIR, 'best_mlp_model.pt')))
    mlp_model.eval()

    test_sets = {
        "mypersonality": (test1_embeddings, test1_labels),
        "Essay": (test2_embeddings, test2_labels)
    }

    with torch.no_grad():
        for split_name, (embeddings, labels) in test_sets.items():
            embeddings = embeddings.to(device)
            outputs = mlp_model(embeddings)
            probs = torch.sigmoid(outputs).cpu()
            binary_preds = (probs > 0.5).int().numpy()
            true_labels = labels.numpy()

            f1_micro_overall = f1_score(true_labels, binary_preds, average='micro')
            accuracy_overall = accuracy_score(true_labels, binary_preds)

            print("\n" + "-"*50)
            print(f"Final Test Set '{split_name}' Evaluation Results (Overall):")
            print(f"  Subset Accuracy (Exact Match): {accuracy_overall:.4f}")
            print(f"  F1 Score (micro): {f1_micro_overall:.4f}")
            
            print("\nPer-Trait Test Results:")
            for i, trait in enumerate(Config.ALL_TARGET_COLUMNS):
                trait_labels = true_labels[:, i]
                trait_preds = binary_preds[:, i]
                trait_acc = accuracy_score(trait_labels, trait_preds)
                trait_f1 = f1_score(trait_labels, trait_preds, average='binary')
                
                print(f"  --- {trait} ---")
                print(f"    Accuracy: {trait_acc:.4f}")
                print(f"    F1 Score: {trait_f1:.4f}")
            print("-"*50)

    print("\n SCRIPT FINISHED ".center(80, "="))
    del model, model_lora, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
